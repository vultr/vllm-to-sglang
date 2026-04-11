FROM lmsysorg/sglang-rocm:v0.5.10rc0-rocm700-mi30x-20260411

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]