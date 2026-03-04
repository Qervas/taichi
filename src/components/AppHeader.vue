<template>
  <header>
    <a href="#" class="site-brand">
      <h1>TAICHI&nbsp;POSTS</h1>
      <span class="site-badge">MLS-MPM · SIGGRAPH 2018</span>
    </a>
    <div class="hdr-right">
      <div class="live-dot"><i></i> 270K particles</div>
      <a href="https://github.com/Qervas/taichi" class="gh-btn" target="_blank" rel="noopener">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.44 9.8 8.21 11.39.6.11.79-.26.79-.58v-2.23c-3.34.73-4.03-1.42-4.03-1.42-.55-1.39-1.34-1.76-1.34-1.76-1.09-.74.08-.73.08-.73 1.2.09 1.84 1.24 1.84 1.24 1.07 1.83 2.81 1.3 3.49 1 .11-.78.42-1.3.76-1.6-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.14-.3-.54-1.52.12-3.18 0 0 1-.32 3.3 1.23a11.5 11.5 0 0 1 6 0c2.3-1.55 3.3-1.23 3.3-1.23.66 1.66.25 2.88.12 3.18.77.84 1.24 1.91 1.24 3.22 0 4.61-2.81 5.63-5.48 5.92.43.37.82 1.1.82 2.22v3.29c0 .32.19.69.8.58C20.56 21.8 24 17.3 24 12c0-6.63-5.37-12-12-12z"/></svg>
        GitHub
      </a>
    </div>
  </header>
</template>

<style scoped>
header {
  display: flex; align-items: center; justify-content: space-between;
  padding: .75rem 2rem;
  border-bottom: 1px solid var(--border);
}
.site-brand {
  display: flex; align-items: center; gap: .6rem;
}
.site-brand h1 {
  font-family: var(--ff-brand); font-weight: 900;
  font-size: .75rem; letter-spacing: .18em; color: #fff;
}
.site-badge {
  font-family: var(--ff-mono); font-size: .52rem;
  padding: .15rem .45rem; border-radius: 3px;
  background: rgba(0,212,255,.08); color: var(--cyan-dim);
  letter-spacing: .04em;
}
.hdr-right { display: flex; align-items: center; gap: .8rem; }
.live-dot {
  display: flex; align-items: center; gap: .3rem;
  font-family: var(--ff-mono); font-size: .55rem; color: var(--text-dim);
}
.live-dot i {
  width: 5px; height: 5px; border-radius: 50%;
  background: #34d399; display: inline-block;
  box-shadow: 0 0 6px #34d399;
  animation: blink 2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }
.gh-btn {
  display: inline-flex; align-items: center; gap: .25rem;
  font-family: var(--ff-mono); font-size: .55rem;
  padding: .2rem .55rem; border-radius: 4px;
  border: 1px solid var(--border);
  color: var(--text-dim);
  transition: border-color .2s, color .2s;
}
.gh-btn:hover { border-color: var(--border-h); color: #fff; }

@media (max-width: 780px) {
  header { padding: 0 1rem; }
  .site-badge, .live-dot { display: none; }
}
</style>
