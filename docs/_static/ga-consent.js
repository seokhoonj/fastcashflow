// The theme injects Google Consent Mode with analytics_storage denied by
// default and exposes no option to change it, so GA4 never sets a client_id
// cookie and the Realtime report stays empty. This public docs site has no
// login or PII, so grant analytics consent. Loaded after the theme's GA
// snippet, this runs default(denied) -> config -> update(granted).
window.dataLayer = window.dataLayer || [];
function gtag() { dataLayer.push(arguments); }
gtag('consent', 'update', {
    'analytics_storage': 'granted'
});
