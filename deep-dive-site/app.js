// 侧边导航高亮 + 更新日期
const links = document.querySelectorAll('.nav a');
const sections = [...links].map(a => document.querySelector(a.getAttribute('href')));

function onScroll() {
  const y = window.scrollY + 120;
  let current = 0;
  sections.forEach((s, i) => { if (s && s.offsetTop <= y) current = i; });
  links.forEach((a, i) => a.classList.toggle('active', i === current));
}
window.addEventListener('scroll', onScroll, { passive: true });
onScroll();

document.getElementById('updated').textContent =
  new Date(document.lastModified).toLocaleDateString('zh-CN');
