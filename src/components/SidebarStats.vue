<template>
  <div class="panel">
    <div class="panel-hdr"><span>// Simulation Scale</span></div>
    <div class="stats">
      <div class="stat" v-for="s in STATS" :key="s.k">
        <span class="stat-k">{{ s.k }}</span>
        <span class="stat-v">{{ s.v }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { STATS } from '../data/posts.js'
</script>

<style scoped>
.panel {
  border: 1px solid var(--border); border-radius: 8px;
  background: var(--bg-card); overflow: hidden;
}
.panel-hdr {
  font-family: var(--ff-mono); font-size: .55rem;
  color: var(--cyan-dim); padding: .4rem .65rem;
  border-bottom: 1px solid var(--border);
  letter-spacing: .04em;
}
.stats { padding: .45rem .65rem; display: grid; gap: .25rem; }
.stat {
  display: flex; justify-content: space-between; align-items: center;
  font-family: var(--ff-mono); font-size: .55rem;
}
.stat-k { color: var(--text-dim); }
.stat-v { color: var(--cyan); font-weight: 500; }
</style>
