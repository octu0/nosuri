// swift-tools-version: 6.0
import PackageDescription

#if os(macOS)
let packageDependencies: [Package.Dependency] = [
    .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.21.0")
]
let engineDependencies: [Target.Dependency] = [
    .product(name: "MLX", package: "mlx-swift"),
    .product(name: "MLXNN", package: "mlx-swift"),
    .product(name: "MLXOptimizers", package: "mlx-swift")
]
let mlxTargetDependencies: [Target.Dependency] = [
    "Nosuri",
    .product(name: "MLX", package: "mlx-swift")
]
#else
let packageDependencies: [Package.Dependency] = []
let engineDependencies: [Target.Dependency] = []
let mlxTargetDependencies: [Target.Dependency] = ["Nosuri"]
#endif

let package = Package(
    name: "nosuri",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "Nosuri", targets: ["Nosuri"]),
        .executable(name: "train", targets: ["train"]),
        .executable(name: "react", targets: ["react"]),
        .executable(name: "server", targets: ["server"]),
        .executable(name: "benchmark", targets: ["benchmark"])
    ],
    dependencies: packageDependencies,
    targets: [
        .target(
            name: "Nosuri",
            dependencies: engineDependencies,
            path: "Sources/Nosuri"
        ),
        .executableTarget(
            name: "train",
            dependencies: mlxTargetDependencies,
            path: "script/train"
        ),
        .executableTarget(
            name: "react",
            dependencies: ["Nosuri"],
            path: "script/react"
        ),
        .executableTarget(
            name: "server",
            dependencies: ["Nosuri"],
            path: "script/server"
        ),
        .executableTarget(
            name: "benchmark",
            dependencies: ["Nosuri"],
            path: "script/benchmark"
        ),
        .testTarget(
            name: "NosuriTests",
            dependencies: mlxTargetDependencies,
            path: "Tests/NosuriTests"
        )
    ]
)
