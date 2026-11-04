include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
    server-tools.cpp
    server-tools.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PUBLIC ../mtmd)
target_include_directories(${TARGET} PRIVATE ../omni)
target_include_directories(${TARGET} PRIVATE ../omni/voxcpm2)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE llama-common omni mtmd voxcpm2_runtime ${CMAKE_THREAD_LIBS_INIT})

# llama-omni-server executable (standalone omni HTTP API)

if (LLAMA_OPENSSL)
    find_package(OpenSSL)
endif()

set(TARGET_OMNI llama-omni-server)

add_executable(${TARGET_OMNI}
    server-omni.cpp
    session.h
    session.cpp
    protocol.h
    protocol.cpp
    ws_handler.h
    ws_handler.cpp
)
install(TARGETS ${TARGET_OMNI} RUNTIME)

target_link_libraries(${TARGET_OMNI} PRIVATE omni llama-common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
if (LLAMA_OPENSSL AND OpenSSL_FOUND)
    target_link_libraries(${TARGET_OMNI} PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(${TARGET_OMNI} PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
endif()
target_compile_features(${TARGET_OMNI} PRIVATE cxx_std_17)
target_include_directories(${TARGET_OMNI} PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(${TARGET_OMNI} PRIVATE ${CMAKE_SOURCE_DIR}/vendor/cpp-httplib)
target_include_directories(${TARGET_OMNI} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)

# llama-tts-server executable (standalone VoxCPM2 TTS HTTP API)

set(TARGET_VOXCPM2 llama-tts-server)

add_executable(${TARGET_VOXCPM2} server-voxcpm2.cpp)
install(TARGETS ${TARGET_VOXCPM2} RUNTIME)

target_link_libraries(${TARGET_VOXCPM2} PRIVATE voxcpm2_runtime llama-common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
if (LLAMA_OPENSSL AND OpenSSL_FOUND)
    target_link_libraries(${TARGET_VOXCPM2} PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(${TARGET_VOXCPM2} PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
endif()
target_compile_features(${TARGET_VOXCPM2} PRIVATE cxx_std_17)
target_include_directories(${TARGET_VOXCPM2} PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(${TARGET_VOXCPM2} PRIVATE ${CMAKE_SOURCE_DIR}/vendor/cpp-httplib)
target_include_directories(${TARGET_VOXCPM2} PRIVATE ${CMAKE_SOURCE_DIR}/vendor)

# Minimal OpenAI-compatible server using the current server runtime

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/ui.h"
    "#pragma once\n"
)

add_executable(moeshift
    server-les.cpp
    server-les-router.cpp
    server-models.cpp
    server-http.cpp
)
set_target_properties(moeshift PROPERTIES OUTPUT_NAME moeshift INSTALL_RPATH "@loader_path/../lib")

target_link_libraries(moeshift PRIVATE
    server-context
    llama-common
    cpp-httplib
    ${CMAKE_THREAD_LIBS_INIT}
)

target_include_directories(moeshift PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/vendor
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_compile_features(moeshift PRIVATE cxx_std_17)

if (APPLE AND GGML_METAL)
    target_sources(moeshift PRIVATE gpu-keepalive.mm gpu-keepalive.h)
    set_source_files_properties(gpu-keepalive.mm PROPERTIES LANGUAGE CXX COMPILE_OPTIONS "-x;objective-c++;-fobjc-arc")
    target_compile_definitions(moeshift PRIVATE LLAMA_LES_METAL_KEEPALIVE)
    target_link_libraries(moeshift PRIVATE "-framework Foundation" "-framework Metal")
endif()

install(TARGETS moeshift RUNTIME DESTINATION bin COMPONENT moeshift)
install(FILES ${CMAKE_SOURCE_DIR}/configs/moeshift.json DESTINATION share/moeshift COMPONENT moeshift)
install(FILES ${CMAKE_SOURCE_DIR}/tools/moe/experts.py ${CMAKE_SOURCE_DIR}/tools/moe/onboarding.py
    DESTINATION share/moeshift/tools/moe COMPONENT moeshift)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/gguf-py/gguf
    DESTINATION share/moeshift/gguf-py COMPONENT moeshift
    FILES_MATCHING PATTERN "*.py" PATTERN "__pycache__" EXCLUDE)

install(FILES ${CMAKE_SOURCE_DIR}/LICENSE
    DESTINATION share/moeshift/licenses COMPONENT moeshift)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/licenses/
    DESTINATION share/moeshift/licenses COMPONENT moeshift)
install(FILES ${CMAKE_SOURCE_DIR}/gguf-py/LICENSE
    DESTINATION share/moeshift/licenses
    RENAME LICENSE-gguf-py COMPONENT moeshift)
