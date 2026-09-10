// Simple script to highlight the current navigation link based on scroll position
const sections = document.querySelectorAll('section');
const navLinks = document.querySelectorAll('nav a');

function onScroll() {
  let scrollPos = window.scrollY + 80; // offset for header height
  sections.forEach(section => {
    if (scrollPos >= section.offsetTop && scrollPos < section.offsetTop + section.offsetHeight) {
      const id = section.getAttribute('id');
      navLinks.forEach(link => {
        link.classList.toggle('active', link.getAttribute('href') === `#${id}`);
      });
    }
  });
}

window.addEventListener('scroll', onScroll);

// Add basic active link styling via JS (could also be done in CSS)
navLinks.forEach(link => {
  link.style.transition = 'color 0.3s';
});
