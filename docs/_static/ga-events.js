// Track clicks on links to the live demo (demo.fastcashflow.org). Both links --
// the navbar external link and the index-page button -- are theme-rendered, so
// we cannot attach an inline onclick; delegate from the document instead. The
// click lands on the docs GA property; the demo site has its own property.
document.addEventListener('click', function (e) {
    var link = e.target.closest('a[href^="https://demo.fastcashflow.org"]');
    if (!link || typeof gtag !== 'function') return;
    var inNav = !!link.closest('.bd-header, header, nav');
    gtag('event', 'demo_click', {
        location: inNav ? 'navbar' : 'hero',
        link_url: link.href
    });
});
