const CACHE_VERSION = Date.now();
const CACHE_NAME = 'edge-crew-' + CACHE_VERSION;

// Install — skip waiting immediately (force activate new SW)
self.addEventListener('install', (event) => {
  self.skipWaiting();
});

// Activate — delete ALL old caches, claim clients
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.map((key) => caches.delete(key)))
    ).then(() => self.clients.claim())
  );
});

// Fetch — NETWORK FIRST for everything. Cache is ONLY offline fallback.
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // API calls: network only, never cache
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(fetch(request));
    return;
  }

  // Everything else: network first, cache fallback (offline only)
  event.respondWith(
    fetch(request)
      .then((response) => {
        // Cache successful responses for offline use
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
        }
        return response;
      })
      .catch(() => caches.match(request))
  );
});
