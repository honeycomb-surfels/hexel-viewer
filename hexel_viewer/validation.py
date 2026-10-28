"""Browser test helpers for authenticated CUDA frame comparisons."""

def capture_render_frames(page: object) -> None:
    """Capture decoded fetch bodies; Chromium's network inspector can evict them."""
    page.add_init_script("""(() => {
        window.hexelsValidationFrames = {};
        const originalFetch = window.fetch;
        window.fetch = async (...args) => {
            const response = await originalFetch(...args);
            if (new URL(response.url).pathname === '/api/render' && response.ok) {
                const payload = JSON.parse(args[1].body);
                const bytes = await response.clone().arrayBuffer();
                window.hexelsValidationFrames[`${payload.mode}/${payload.compare_mode}`] = {
                    payload, bytes: Array.from(new Uint8Array(bytes)),
                    stats: JSON.parse(response.headers.get('X-Hexels-Stats')),
                };
            }
            return response;
        };
    })();""")
