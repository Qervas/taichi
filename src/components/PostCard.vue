<template>
  <div
    class="card"
    :class="post.categories.map(c => 'c-' + c)"
    :style="{ animationDelay: index * 0.07 + 's' }"
    :id="'post-' + post.id"
  >
    <!-- Single video thumbnail -->
    <div v-if="post.video" class="vid-thumb" @click="$emit('open-video', post.title, post.video)" role="button" tabindex="0" :aria-label="'Play ' + post.title">
      <img :src="post.thumb" alt="" loading="lazy">
      <div class="vid-play-wrap">
        <div class="vid-play-btn">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21"/></svg>
        </div>
        <span class="vid-play-label">Play video</span>
      </div>
    </div>
    <!-- Multiple video thumbnails -->
    <div v-if="post.videos" class="vid-row">
      <div
        v-for="v in post.videos" :key="v.src"
        class="vid-thumb vid-thumb--half"
        @click="$emit('open-video', v.label, v.src)"
        role="button" tabindex="0" :aria-label="'Play ' + v.label"
      >
        <img :src="v.thumb" alt="" loading="lazy">
        <div class="vid-play-wrap">
          <div class="vid-play-btn">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21"/></svg>
          </div>
          <span class="vid-play-label">{{ v.label }}</span>
        </div>
      </div>
    </div>

    <!-- Body -->
    <div class="card-body">
      <div class="card-meta">
        <span class="phase-badge">{{ post.phase }}</span>
        <span class="post-date">{{ fmtDate(post.date) }}</span>
        <div class="card-meta-right">
          <div class="tags">
            <span v-for="c in post.categories" :key="c" class="tag" :class="c">
              <span v-html="ICONS[c]"></span>{{ c }}
            </span>
          </div>
          <button class="share-btn" @click="$emit('share', post)" title="Copy link">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>
          </button>
        </div>
      </div>
      <h2 class="card-title">{{ post.title }}</h2>
      <p class="card-sub">{{ post.subtitle }}</p>
      <p class="card-desc">{{ post.description }}</p>
      <div v-if="post.slides || post.code || post.refs || post.video || post.videos" class="card-links">
        <button v-if="post.slides" class="lnk lnk-slides" @click="$emit('open-slides', post)">
          <span v-html="ICONS.slides"></span> View Slides
        </button>
        <button v-if="post.code" class="lnk lnk-code" @click="$emit('open-code', post)">
          <span v-html="ICONS.code"></span> View Code
        </button>
        <a v-for="r in post.refs" :key="r.url" :href="r.url" class="lnk lnk-ref" target="_blank" rel="noopener">
          <span v-html="ICONS.ref"></span> {{ r.label }}
        </a>

        <!-- Download buttons -->
        <a v-if="post.video" :href="post.video" :download="post.phase + ' - ' + post.title + '.mp4'" class="lnk lnk-dl-video">
          <span v-html="ICONS.download"></span> Video
        </a>
        <a v-for="v in post.videos" :key="'dl-' + v.src" :href="v.src" :download="post.phase + ' - ' + v.label + '.mp4'" class="lnk lnk-dl-video">
          <span v-html="ICONS.download"></span> {{ v.label }}
        </a>

        <button v-if="post.code && !post.codeFiles" class="lnk lnk-dl-code" @click="downloadSingleCode(post)">
          <span v-html="ICONS.download"></span> Code
        </button>
        <button v-if="post.codeFiles" class="lnk lnk-dl-code" :disabled="zipping" @click="downloadCodeZip(post)">
          <span v-html="ICONS.download"></span> {{ zipping ? 'Zipping…' : 'Code (.zip)' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { ICONS, fmtDate } from '../data/posts.js'

defineProps({
  post: { type: Object, required: true },
  index: { type: Number, default: 0 },
})
defineEmits(['open-slides', 'open-video', 'open-code', 'share'])

const zipping = ref(false)

function downloadSingleCode(post) {
  const a = document.createElement('a')
  a.href = post.code
  a.download = post.code.split('/').pop()
  a.click()
}

async function downloadCodeZip(post) {
  if (zipping.value) return
  zipping.value = true
  try {
    const { default: JSZip } = await import('jszip')
    const zip = new JSZip()
    for (const file of post.codeFiles) {
      const url = `${post.code}/${file}`
      const resp = await fetch(url)
      const text = await resp.text()
      zip.file(file, text)
    }
    const blob = await zip.generateAsync({ type: 'blob' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `${post.phase} - ${post.title}.zip`
    a.click()
    URL.revokeObjectURL(a.href)
  } finally {
    zipping.value = false
  }
}
</script>

<style scoped>
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px; overflow: hidden;
  animation: fadeUp .35s ease both;
}
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}
.card :deep(.card--highlight) { box-shadow: 0 0 0 2px var(--cyan), 0 0 20px rgba(0,212,255,.18); }

/* ── Thumbnail / Video ── */
.vid-thumb {
  position: relative; cursor: pointer; overflow: hidden;
  aspect-ratio: 16/9; background: #000;
}
.vid-thumb img { width: 100%; height: 100%; object-fit: cover; transition: transform .35s; }
.vid-thumb:hover img { transform: scale(1.04); }
.vid-play-wrap {
  position: absolute; inset: 0;
  display: flex; flex-direction: column; justify-content: center; align-items: center;
  background: rgba(0,0,0,.35);
  opacity: 0; transition: opacity .25s;
}
.vid-thumb:hover .vid-play-wrap { opacity: 1; }
.vid-play-btn {
  width: 42px; height: 42px; border-radius: 50%;
  background: rgba(0,212,255,.15); border: 1.5px solid var(--cyan);
  display: flex; align-items: center; justify-content: center;
  color: var(--cyan); backdrop-filter: blur(4px);
}
.vid-play-label {
  font-family: var(--ff-mono); font-size: .5rem;
  color: rgba(255,255,255,.8); margin-top: .3rem; letter-spacing: .04em;
}
.vid-row { display: grid; grid-template-columns: 1fr 1fr; }
.vid-thumb--half { aspect-ratio: 16/9; }

/* ── Body ── */
.card-body { padding: .7rem .85rem .8rem; }
.card-meta {
  display: flex; align-items: center; gap: .4rem;
  flex-wrap: wrap; margin-bottom: .3rem;
}
.card-meta-right {
  margin-left: auto;
  display: flex; align-items: center; gap: .3rem;
}
.phase-badge {
  font-family: var(--ff-brand); font-size: .45rem;
  padding: .12rem .4rem; border-radius: 3px;
  background: rgba(0,212,255,.1); color: var(--cyan);
  letter-spacing: .1em; text-transform: uppercase;
}
.post-date {
  font-family: var(--ff-mono); font-size: .48rem; color: var(--text-dim);
}
.tags { display: flex; gap: .25rem; }
.tag {
  display: inline-flex; align-items: center; gap: .2rem;
  font-family: var(--ff-mono); font-size: .45rem;
  padding: .08rem .35rem; border-radius: 3px;
  text-transform: uppercase; letter-spacing: .04em;
}
.tag.slides { background: rgba(167,139,250,.1); color: #a78bfa; }
.tag.code   { background: rgba(52,211,153,.1);  color: #34d399; }
.tag.vids,
.tag.video  { background: rgba(244,114,182,.1); color: #f472b6; }

.share-btn {
  background: none; border: none; color: var(--text-dim);
  cursor: pointer; padding: .15rem; border-radius: 4px;
  transition: color .15s, background .15s;
}
.share-btn:hover { color: var(--cyan); background: rgba(0,212,255,.06); }

.card-title {
  font-family: var(--ff-head); font-weight: 700;
  font-size: .78rem; line-height: 1.35; margin-bottom: .15rem;
}
.card-sub {
  font-family: var(--ff-mono); font-size: .52rem;
  color: var(--cyan-dim); margin-bottom: .3rem;
}
.card-desc {
  font-size: .7rem; color: var(--text-dim); line-height: 1.55;
}
.card-links {
  display: flex; gap: .35rem; flex-wrap: wrap; margin-top: .5rem;
}
.lnk {
  display: inline-flex; align-items: center; gap: .25rem;
  font-family: var(--ff-mono); font-size: .5rem;
  padding: .2rem .5rem; border-radius: 5px;
  cursor: pointer; border: 1px solid transparent;
  transition: background .15s, border-color .15s;
}
.lnk-slides { background: rgba(167,139,250,.08); color: #a78bfa; border-color: rgba(167,139,250,.18); }
.lnk-slides:hover { background: rgba(167,139,250,.16); }
.lnk-code   { background: rgba(52,211,153,.08);  color: #34d399; border-color: rgba(52,211,153,.18); }
.lnk-code:hover   { background: rgba(52,211,153,.16); }
.lnk-ref    { background: rgba(255,255,255,.04); color: var(--text-dim); border-color: var(--border); }
.lnk-ref:hover { background: rgba(255,255,255,.08); color: #fff; }

.lnk-dl-video { background: rgba(251,191,36,.08); color: #fbbf24; border-color: rgba(251,191,36,.18); text-decoration: none; }
.lnk-dl-video:hover { background: rgba(251,191,36,.16); }
.lnk-dl-code  { background: rgba(52,211,153,.08); color: #34d399; border-color: rgba(52,211,153,.18); }
.lnk-dl-code:hover  { background: rgba(52,211,153,.16); }
.lnk-dl-code:disabled { opacity: .6; cursor: wait; }
</style>
