<template>
  <header>
    <div class="hdr-left">
      <a href="#" class="site-brand" @click.prevent="$emit('update:view', 'posts')">
        <h1>TAICHI&nbsp;POSTS</h1>
      </a>
      <div class="view-toggle">
        <button class="vt-pill" :class="{ active: view === 'posts' }" @click="$emit('update:view', 'posts')">
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
          Posts
        </button>
        <button class="vt-pill" :class="{ active: view === 'code' }" @click="$emit('update:view', 'code')">
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
          Codebase
        </button>
      </div>
    </div>
    <div class="hdr-right">
      <span class="breath-dot"></span>
      <a href="https://github.com/Qervas/taichi" class="gh-btn" target="_blank" rel="noopener">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.44 9.8 8.21 11.39.6.11.79-.26.79-.58v-2.23c-3.34.73-4.03-1.42-4.03-1.42-.55-1.39-1.34-1.76-1.34-1.76-1.09-.74.08-.73.08-.73 1.2.09 1.84 1.24 1.84 1.24 1.07 1.83 2.81 1.3 3.49 1 .11-.78.42-1.3.76-1.6-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.14-.3-.54-1.52.12-3.18 0 0 1-.32 3.3 1.23a11.5 11.5 0 0 1 6 0c2.3-1.55 3.3-1.23 3.3-1.23.66 1.66.25 2.88.12 3.18.77.84 1.24 1.91 1.24 3.22 0 4.61-2.81 5.63-5.48 5.92.43.37.82 1.1.82 2.22v3.29c0 .32.19.69.8.58C20.56 21.8 24 17.3 24 12c0-6.63-5.37-12-12-12z"/></svg>
        GitHub
      </a>
    </div>
  </header>
</template>

<script setup>
defineProps({
  view: { type: String, default: 'posts' },
})
defineEmits(['update:view'])
</script>

<style scoped>
header {
  display: flex; align-items: center; justify-content: space-between;
  padding: .5rem 2rem;
  border-bottom: 1px solid var(--border);
  min-height: 44px;
}
.hdr-left {
  display: flex; align-items: center; gap: .8rem;
}
.site-brand {
  display: flex; align-items: center; gap: .6rem;
}
.site-brand h1 {
  font-family: var(--ff-brand); font-weight: 900;
  font-size: .75rem; letter-spacing: .18em; color: #fff;
}

/* ── View toggle pills ── */
.view-toggle {
  display: flex; gap: .15rem;
  background: rgba(255,255,255,.04);
  border-radius: 6px; padding: .12rem;
}
.vt-pill {
  display: inline-flex; align-items: center; gap: .25rem;
  font-family: var(--ff-mono); font-size: .5rem;
  padding: .22rem .55rem; border-radius: 4px;
  background: transparent; border: none;
  color: var(--text-dim); cursor: pointer;
  transition: background .15s, color .15s;
}
.vt-pill:hover { color: rgba(255,255,255,.8); }
.vt-pill.active {
  background: rgba(167,139,250,.15); color: #a78bfa;
}

.hdr-right { display: flex; align-items: center; gap: .8rem; }
.breath-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--cyan);
  box-shadow: 0 0 8px var(--cyan), 0 0 20px rgba(0,212,255,.25);
  animation: breathe 3s ease-in-out infinite;
}
@keyframes breathe {
  0%, 100% { opacity: 1; box-shadow: 0 0 8px var(--cyan), 0 0 20px rgba(0,212,255,.25); }
  50% { opacity: .3; box-shadow: 0 0 4px var(--cyan), 0 0 8px rgba(0,212,255,.1); }
}
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
  header { padding: .4rem 1rem; }
  .breath-dot { display: none; }
  .site-brand h1 { font-size: .6rem; }
  .vt-pill { font-size: .45rem; padding: .18rem .4rem; }
}
</style>
