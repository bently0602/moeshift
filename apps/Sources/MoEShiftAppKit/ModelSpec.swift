import Foundation

/// One file of a model's GGUF on Hugging Face, with the size and SHA-256 Hugging Face reports for it.
public struct ModelFile: Sendable {
    public let remotePath: String
    public let size: Int64
    public let sha256: String

    public var name: String { (remotePath as NSString).lastPathComponent }
}

/// Everything an app needs to know about its model. The tuned settings are not here: they come from
/// the bundled copy of configs/moeshift.json, the profile named `alias`.
public struct ModelSpec: Sendable {
    public let alias: String
    public let displayName: String
    public let repo: String
    public let files: [ModelFile]
    /// Bytes of the prepared `experts/` folder.
    public let expertBytes: Int64
    /// The GGUF without expert tensors that `moeshift experts strip` writes.
    public let strippedName: String
    public let defaultPort: Int
    public let tagline: String
    /// Thinking for requests that set no reasoning effort, until the user picks one.
    public let defaultReasoningEffort: String

    public var downloadBytes: Int64 { files.reduce(0) { $0 + $1.size } }
    /// Approximate size of the stripped GGUF: everything but the experts.
    public var strippedBytes: Int64 { downloadBytes - expertBytes }
    /// What stays in the model folder after preparing.
    public var keptBytes: Int64 { expertBytes + strippedBytes }

    public func url(for file: ModelFile) -> URL {
        URL(string: "https://huggingface.co/\(repo)/resolve/main/\(file.remotePath)?download=true")!
    }

    public static let flashNext = ModelSpec(
        alias: "qwen-3.8-flash-next-rco-gsq-iq3-xxs",
        displayName: "Qwen 3.8 Flash Next",
        repo: "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        files: [
            ModelFile(remotePath: "IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
                      size: 47_039_860_096,
                      sha256: "219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15"),
            ModelFile(remotePath: "IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf",
                      size: 28_800_138_432,
                      sha256: "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
        ],
        expertBytes: 42_912_972_800,
        strippedName: "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-les.gguf",
        defaultPort: 8080,
        tagline: "GSQ-RCO IQ3_XXS",
        // Gave the unlimited output on all 16 measured problems and caps runaway thinking.
        defaultReasoningEffort: "medium")

    /// Flash Next with half of its routed experts removed (256 of 512 per layer), chosen for code,
    /// agentic tool use and vision. Shard 2, the per-layer n-gram table, is byte for byte the same
    /// as Flash Next's.
    public static let flashNextCoder = ModelSpec(
        alias: "qwen-3.8-flash-next-coder-rco-gsq-iq1-m",
        displayName: "Qwen 3.8 Flash Next Coder",
        repo: "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF",
        files: [
            ModelFile(remotePath: "IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf",
                      size: 29_608_446_496,
                      sha256: "bfa3644db4adfff072fc946c20994f2909d2cf89879d83f80a4c811d4016c211"),
            ModelFile(remotePath: "IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf",
                      size: 28_800_138_432,
                      sha256: "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
        ],
        expertBytes: 25_146_163_200,
        strippedName: "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-les.gguf",
        defaultPort: 8082,
        tagline: "GSQ-RCO Coder, 256 of 512 experts",
        defaultReasoningEffort: "medium")

    public static let a3b = ModelSpec(
        alias: "qwen-3.6-35b-a3b-q6-xl",
        displayName: "Qwen3.6 35B A3B",
        repo: "unsloth/Qwen3.6-35B-A3B-GGUF",
        files: [
            ModelFile(remotePath: "Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf",
                      size: 31_843_777_504,
                      sha256: "f6b6c6d5cfa6f00d964eeb7add28eb14ce7481734d506b90681007678cd2c484"),
        ],
        expertBytes: 29_154_607_104,
        strippedName: "Qwen3.6-35B-A3B-UD-Q6_K_XL-les.gguf",
        defaultPort: 8081,
        tagline: "unsloth UD-Q6_K_XL",
        defaultReasoningEffort: "high")
}
