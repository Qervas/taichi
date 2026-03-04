<template>
  <AppHeader />

  <div class="wrap">
    <aside class="sidebar">
      <SidebarCalendar v-model:activeDate="activeDate" />
      <SidebarStats />
    </aside>

    <main class="main">
      <FilterBar v-model:activeCat="activeCat" :resultsMeta="resultsMeta" />

      <div class="posts">
        <TransitionGroup name="fade">
          <PostCard
            v-for="(post, idx) in filteredPosts" :key="post.id"
            :post="post"
            :index="idx"
            @open-slides="openSlides"
            @open-video="openVideo"
            @open-code="openCode"
            @share="sharePost"
          />
        </TransitionGroup>
      </div>

      <div v-if="filteredPosts.length === 0" class="empty">
        <div class="empty-icon">&#9670;</div>
        No posts match the current filter.
      </div>
    </main>
  </div>

  <SlidesModal
    :open="modal.open"
    :title="modal.title"
    :url="modal.url"
    @close="closeModal"
  />

  <VideoModal
    :open="videoModal.open"
    :title="videoModal.title"
    :url="videoModal.url"
    @close="closeVideo"
  />

  <CodeModal
    :open="codeModal.open"
    :post="codeModal.post"
    @close="closeCode"
  />

  <Transition name="toast">
    <div v-if="toast" class="toast">Link copied!</div>
  </Transition>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onUnmounted, nextTick } from 'vue'
import { POSTS, fmtDate } from './data/posts.js'
import AppHeader from './components/AppHeader.vue'
import SidebarCalendar from './components/SidebarCalendar.vue'
import SidebarStats from './components/SidebarStats.vue'
import FilterBar from './components/FilterBar.vue'
import PostCard from './components/PostCard.vue'
import SlidesModal from './components/SlidesModal.vue'
import VideoModal from './components/VideoModal.vue'
import CodeModal from './components/CodeModal.vue'

/* ── State ── */
const activeCat = ref('all')
const activeDate = ref(null)

const modal = reactive({ open: false, title: '', url: '' })
const videoModal = reactive({ open: false, title: '', url: '' })
const codeModal = reactive({ open: false, post: null })

const toast = ref(false)
let _toastTimer = null

/* ── Filtered posts ── */
const filteredPosts = computed(() =>
  POSTS
    .filter(p =>
      (activeCat.value === 'all' || p.categories.includes(activeCat.value)) &&
      (!activeDate.value || p.date === activeDate.value)
    )
    .sort((a, b) => b.date.localeCompare(a.date))
)

const resultsMeta = computed(() => {
  const n = filteredPosts.value.length
  if (!n) return ''
  let m = `${n} post${n !== 1 ? 's' : ''}`
  if (activeDate.value) m += ` · ${fmtDate(activeDate.value)}`
  if (activeCat.value !== 'all') m += ` · #${activeCat.value}`
  return m
})

/* ── Slides modal ── */
function openSlides(post) {
  modal.title = `${post.phase} — ${post.title}`
  modal.url = post.slides
  modal.open = true
}
function closeModal() {
  modal.open = false
  modal.url = ''
  if (document.fullscreenElement) document.exitFullscreen()
}

/* ── Video modal ── */
function openVideo(title, url) {
  videoModal.title = title
  videoModal.url = url
  videoModal.open = true
}
function closeVideo() {
  videoModal.open = false
  videoModal.url = ''
}

/* ── Code modal ── */
function openCode(post) {
  codeModal.post = post
  codeModal.open = true
}
function closeCode() {
  codeModal.open = false
  codeModal.post = null
}

/* ── Share ── */
function sharePost(post) {
  const url = `${location.origin}${location.pathname}#post-${post.id}`
  navigator.clipboard.writeText(url).then(() => {
    clearTimeout(_toastTimer)
    toast.value = true
    _toastTimer = setTimeout(() => { toast.value = false }, 2000)
  })
}

/* ── ESC key ── */
function onKey(e) {
  if (e.key === 'Escape') {
    if (codeModal.open) closeCode()
    else if (videoModal.open) closeVideo()
    else closeModal()
  }
}

onMounted(() => {
  // Hide page loader
  const loader = document.getElementById('loader')
  if (loader) loader.classList.add('hidden')

  window.addEventListener('keydown', onKey)

  // Scroll to linked post if URL has #post-{id}
  const hash = location.hash
  if (hash && hash.startsWith('#post-')) {
    nextTick(() => {
      const el = document.getElementById(hash.slice(1))
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' })
        el.classList.add('card--highlight')
        setTimeout(() => el.classList.remove('card--highlight'), 2000)
      }
    })
  }
})
onUnmounted(() => window.removeEventListener('keydown', onKey))
</script>

<style scoped>
.wrap {
  display: grid; grid-template-columns: 240px 1fr;
  gap: 1.2rem; max-width: 1060px;
  margin: 0 auto; padding: 1.2rem 2rem;
}
.sidebar { position: sticky; top: 1rem; align-self: start; display: grid; gap: .8rem; }
.main { min-width: 0; }
.posts { display: grid; gap: .8rem; }

/* ── Empty state ── */
.empty {
  text-align: center; padding: 2.5rem 1rem;
  color: var(--text-dim); font-family: var(--ff-mono); font-size: .6rem;
}
.empty-icon { font-size: 1.5rem; margin-bottom: .4rem; color: var(--cyan-dim); }

/* ── Toast ── */
.toast {
  position: fixed; bottom: 1.5rem; left: 50%; transform: translateX(-50%);
  background: var(--cyan); color: #000;
  font-family: var(--ff-mono); font-size: .55rem; font-weight: 600;
  padding: .4rem 1rem; border-radius: 6px;
  box-shadow: 0 4px 20px rgba(0,212,255,.3);
  z-index: 9999;
}

@media (max-width: 780px) {
  .wrap { grid-template-columns: 1fr; padding: 1rem; }
  .sidebar { position: static; }
}
</style>
