const CACHE_NAME = 'edge-crew-v1';

// Install — cache shell assets, skip waiting
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) =>
      cache.addAll([
        '/',
        '/static/img/miami-skyline.jpg',
        '/manifest.json'
      ]).catch(() => {})
    )
  );
  self.skipWaiting();
});

// Activate — delete old caches, claim clients
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// Fetch — API calls: network only. Everything else: network first, cache fallback.
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // API calls and POST requests: network only, never cache
  if (url.pathname.startsWith('/api/') || event.request.method !== 'GET') {
    return;
  }

  // Everything else: network first, cache fallback for offline
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        if (response.ok && response.type === 'basic') {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        }
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});
