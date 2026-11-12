// C interface of libzyth_native (crates/zyth-native): run a Zyth program, get its output.
#include <stddef.h>
#include <stdint.h>
// Runs `src` (UTF-8, `len` bytes) as a program. Returns a buffer: a little-endian u64 length,
// then that many bytes of JSON `{ "ok": bool, "output": "...", "error": "..." }`.
uint8_t *zyth_run(const uint8_t *src, size_t len);
// Frees a buffer returned by zyth_run.
void zyth_buf_free(uint8_t *buf);
