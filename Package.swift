// swift-tools-version: 6.0
import PackageDescription

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
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.33.0")
    ],
    targets: [
        .target(
            name: "Nosuri",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift")
            ],
            path: "Sources/Nosuri"
        ),
        .executableTarget(
            name: "train",
            dependencies: [
                "Nosuri",
                .product(name: "MLX", package: "mlx-swift")
            ],
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
            dependencies: [
                "Nosuri",
                .product(name: "MLX", package: "mlx-swift")
            ],
            path: "Tests/NosuriTests"
        )
    ]
)
