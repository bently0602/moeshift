#include "llama-ssd-rows.h"
#include "ggml.h"
#include "ggml-quants.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fcntl.h>
#include <stdexcept>
#include <unistd.h>
#include <vector>

int main() {
    ggml_init_params ip{1024*1024, nullptr, true};
    auto * ctx = ggml_init(ip);
    // Quantized values include all four codes, including +2, and multiple scale blocks.
    block_q2_0 blocks[2]{};
    for (int b=0;b<2;++b) {
        blocks[b].d=ggml_fp32_to_fp16(float(b+1));
        for (auto &q:blocks[b].qs) q=0xe4;
    }
    float qout[128];
    dequantize_row_q2_0(blocks,qout,128);
    for (int i=0;i<128;++i) assert(qout[i]==float((i%4)-1)*(i/64+1));

    char path[]="ssd-rows-test-XXXXXX";
    int fd=mkstemp(path);assert(fd>=0);
    constexpr int cols=160, rows=4000;
    constexpr size_t offset=17;
    std::vector<float> src(cols*rows);
    for (size_t i=0;i<src.size();++i) src[i]=float(i%12345)*0.25f;
    char prefix[offset]{};
    assert(write(fd,prefix,offset)==offset);
    assert(write(fd,src.data(),src.size()*sizeof(float))==(ssize_t)(src.size()*sizeof(float)));
    {
        llama_ssd_rows reader(fd,offset,GGML_TYPE_F32,cols,rows,32768);
        int32_t ids[]={0,25,26,rows-1,25,300,900,25};
        std::vector<float> out(cols*8);
        reader.read_rows(ids,8,out.data());
        for(int n=0;n<8;++n) for(int j=0;j<cols;++j) assert(out[n*cols+j]==src[ids[n]*cols+j]);
        assert(reader.cached_bytes()<=32768);
        auto read=reader.bytes_read();
        reader.read_rows(ids+7,1,out.data());
        assert(reader.bytes_read()==read);
        // More pages than the cache holds in one call: rows are loaded in concurrent passes.
        std::vector<int32_t> many(3000);
        for (size_t i=0;i<many.size();++i) many[i]=int32_t((i*2654435761u)%rows);
        std::vector<float> wide(cols*many.size());
        reader.read_rows(many.data(),many.size(),wide.data());
        for(size_t n=0;n<many.size();++n) for(int j=0;j<cols;++j) assert(wide[n*cols+j]==src[many[n]*cols+j]);
        assert(reader.cached_bytes()<=32768);
        for(int32_t bad:{-1,rows}) {
            bool threw=false;try{reader.read_rows(&bad,1,out.data());}catch(const std::out_of_range&){threw=true;}
            assert(threw);
        }
    }
    {
        // A one-page cache cannot hold a row that crosses a page seam; it is read sequentially.
        llama_ssd_rows reader(fd,offset,GGML_TYPE_F32,cols,rows,16384);
        int32_t ids[]={25,24,25,26};
        std::vector<float> out(cols*4);
        reader.read_rows(ids,4,out.data());
        for(int n=0;n<4;++n) for(int j=0;j<cols;++j) assert(out[n*cols+j]==src[ids[n]*cols+j]);
        assert(reader.cached_bytes()<=16384);
    }
    // Reject declared payload past EOF; an early EOF after opening must also be detected.
    bool threw=false;
    try{llama_ssd_rows bad(fd,offset,GGML_TYPE_F32,cols,rows+1,32768);}catch(const std::exception&){threw=true;}
    assert(threw);
    {
        llama_ssd_rows reader(fd,offset,GGML_TYPE_F32,cols,rows,32768);
        assert(ftruncate(fd,offset+100)==0);
        int32_t id=rows-1;float out[cols];threw=false;
        try{reader.read_rows(&id,1,out);}catch(const std::exception&){threw=true;}
        assert(threw);
    }
    // IQ4_NL is the actual PLE format: check nibble order, scales and page seams.
    {
        assert(ftruncate(fd, 0) == 0);
        assert(lseek(fd, 0, SEEK_SET) == 0);
        assert(write(fd, prefix, offset) == offset);
        std::vector<block_iq4_nl> blocks(rows * cols / 32);
        const int values[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
        for (size_t b=0;b<blocks.size();++b) {
            blocks[b].d=ggml_fp32_to_fp16(float(b%4+1));
            for (int j=0;j<16;++j) blocks[b].qs[j]=(j&15)|(((j+3)&15)<<4);
        }
        const size_t bytes=blocks.size()*sizeof(block_iq4_nl);
        assert(write(fd,blocks.data(),bytes)==(ssize_t)bytes);
        llama_ssd_rows reader(fd,offset,GGML_TYPE_IQ4_NL,cols,rows,32768);
        int32_t ids[]={181,182,364,3999,0,182};
        std::vector<float> out(6*cols);
        reader.read_rows(ids,6,out.data());
        for(int i=0;i<6;++i) for(int j=0;j<cols;++j) {
            int b=ids[i]*(cols/32)+j/32, q=j%32;
            float expected=float(b%4+1)*values[q<16?q:(q-16+3)%16];
            assert(out[i*cols+j]==expected);
        }
        assert(reader.cached_bytes()<=32768);
    }
    close(fd);unlink(path);ggml_free(ctx);
    puts("SSD row bounds, page crossings, eviction, concurrent passes, repeated reads, truncated files and Q2_0 values: PASS");
}
