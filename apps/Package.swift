// swift-tools-version:5.10
// The MoEShift apps: one window per model that downloads and prepares the model, then starts and
// stops the bundled engine. Build both with ./build.sh (no Xcode needed); see README.md.
import PackageDescription

let package = Package(
    name: "MoEShiftApps",
    platforms: [.macOS(.v14)],
    targets: [
        .target(name: "MoEShiftAppKit"),
        .executableTarget(name: "FlashNextApp", dependencies: ["MoEShiftAppKit"]),
        .executableTarget(name: "A3BApp", dependencies: ["MoEShiftAppKit"]),
    ]
)
