FROM swift:6.0-jammy AS builder
WORKDIR /app

COPY Package.swift ./
COPY Sources ./Sources
COPY script ./script

RUN swift build -c release --product server

FROM swift:6.0-jammy-slim
WORKDIR /app

COPY --from=builder /app/.build/release/server /app/server
COPY assets ./assets
COPY .tmp/public ./.tmp/public
COPY .tmp/resources ./.tmp/resources
COPY .tmp/weights.safetensors ./.tmp/weights.safetensors

ENV PORT=8080
ENV WEIGHTS_PATH=/app/.tmp/weights.safetensors
ENV RESOURCES_DIR=/app/.tmp/resources
ENV WEB_DIR=/app/.tmp/public

EXPOSE 8080
CMD ["/app/server"]
