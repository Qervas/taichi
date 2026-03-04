<template>
  <div class="filter-bar">
    <span class="filter-lbl">Filter:</span>
    <button
      v-for="cat in CATS" :key="cat.id"
      class="pill"
      :class="{ on: activeCat === cat.id }"
      :data-c="cat.id"
      @click="$emit('update:activeCat', cat.id)"
    >{{ cat.label }}</button>
  </div>
  <div class="results-meta">{{ resultsMeta }}</div>
</template>

<script setup>
import { CATS } from '../data/posts.js'

defineProps({
  activeCat: { type: String, required: true },
  resultsMeta: { type: String, default: '' },
})
defineEmits(['update:activeCat'])
</script>

<style scoped>
.filter-bar {
  display: flex; align-items: center; gap: .35rem;
  margin-bottom: .6rem; flex-wrap: wrap;
}
.filter-lbl {
  font-family: var(--ff-mono); font-size: .55rem; color: var(--text-dim);
  margin-right: .15rem;
}
.pill {
  font-family: var(--ff-mono); font-size: .52rem;
  padding: .18rem .55rem; border-radius: 4px;
  background: rgba(255,255,255,.04); border: 1px solid var(--border);
  color: var(--text-dim); cursor: pointer;
  transition: border-color .15s, background .15s, color .15s;
}
.pill:hover { border-color: var(--border-h); color: #fff; }
.pill.on {
  background: rgba(0,212,255,.12); border-color: var(--cyan);
  color: var(--cyan); font-weight: 500;
}
.results-meta {
  font-family: var(--ff-mono); font-size: .5rem; color: var(--text-dim);
  margin-bottom: .6rem;
}
</style>
