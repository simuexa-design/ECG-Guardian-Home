/**
 * ECG Guardian Presentation Deck Engine
 * Interactive Keyboard, Touch, Navigation & Print Controls
 */

document.addEventListener('DOMContentLoaded', () => {
  const slides = Array.from(document.querySelectorAll('.slide'));
  const totalSlides = slides.length;
  let currentSlideIndex = 0;

  const progressBar = document.getElementById('progressBar');
  const slideCounterCurrent = document.getElementById('currentSlideNum');
  const slideCounterTotal = document.getElementById('totalSlidesNum');
  const slideCategoryElem = document.getElementById('headerCategory');

  const btnPrev = document.getElementById('btnPrev');
  const btnNext = document.getElementById('btnNext');
  const btnOverview = document.getElementById('btnOverview');
  const btnFullscreen = document.getElementById('btnFullscreen');
  const btnPrint = document.getElementById('btnPrint');

  const overviewModal = document.getElementById('overviewModal');
  const closeOverviewBtn = document.getElementById('closeOverview');
  const overviewGrid = document.getElementById('overviewGrid');

  // Initialize total slide count display
  if (slideCounterTotal) {
    slideCounterTotal.textContent = String(totalSlides).padStart(2, '0');
  }

  // Build Overview Thumbnails
  if (overviewGrid) {
    slides.forEach((slide, index) => {
      const titleElem = slide.querySelector('.slide-title') || slide.querySelector('.main-title');
      const catElem = slide.querySelector('.slide-category') || slide.querySelector('.badge-research');
      const titleText = titleElem ? titleElem.innerText.replace(/\s+/g, ' ').trim() : `Slide ${index + 1}`;
      const catText = catElem ? catElem.innerText.trim() : 'Overview';

      const thumb = document.createElement('div');
      thumb.className = `thumb-card ${index === 0 ? 'active' : ''}`;
      thumb.innerHTML = `
        <div class="thumb-num">SLIDE ${String(index + 1).padStart(2, '0')} · ${catText}</div>
        <div class="thumb-title">${titleText}</div>
      `;
      thumb.addEventListener('click', () => {
        goToSlide(index);
        closeOverview();
      });
      overviewGrid.appendChild(thumb);
    });
  }

  // Slide Navigation Function
  function goToSlide(newIndex) {
    if (newIndex < 0) newIndex = 0;
    if (newIndex >= totalSlides) newIndex = totalSlides - 1;

    slides.forEach((slide, i) => {
      slide.classList.remove('active', 'prev');
      if (i === newIndex) {
        slide.classList.add('active');
      } else if (i < newIndex) {
        slide.classList.add('prev');
      }
    });

    currentSlideIndex = newIndex;

    // Update Counter
    if (slideCounterCurrent) {
      slideCounterCurrent.textContent = String(currentSlideIndex + 1).padStart(2, '0');
    }

    // Update Progress Bar
    if (progressBar) {
      const progressPercent = ((currentSlideIndex + 1) / totalSlides) * 100;
      progressBar.style.width = `${progressPercent}%`;
    }

    // Update Header Category
    const activeSlide = slides[currentSlideIndex];
    if (activeSlide && slideCategoryElem) {
      const cat = activeSlide.dataset.category || 'Clinical Telemetry Deck';
      slideCategoryElem.textContent = cat.toUpperCase();
    }

    // Update Overview Active Thumbnail
    if (overviewGrid) {
      const thumbs = overviewGrid.querySelectorAll('.thumb-card');
      thumbs.forEach((t, i) => {
        t.classList.toggle('active', i === currentSlideIndex);
      });
    }

    // Update URL hash without scroll
    history.replaceState(null, '', `#slide-${currentSlideIndex + 1}`);

    // Update Prev/Next button states
    if (btnPrev) btnPrev.disabled = currentSlideIndex === 0;
    if (btnNext) btnNext.disabled = currentSlideIndex === totalSlides - 1;
  }

  function nextSlide() {
    if (currentSlideIndex < totalSlides - 1) {
      goToSlide(currentSlideIndex + 1);
    }
  }

  function prevSlide() {
    if (currentSlideIndex > 0) {
      goToSlide(currentSlideIndex - 1);
    }
  }

  function toggleOverview() {
    if (overviewModal) {
      overviewModal.classList.toggle('open');
    }
  }

  function closeOverview() {
    if (overviewModal) {
      overviewModal.classList.remove('open');
    }
  }

  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      }
    }
  }

  // Event Listeners
  if (btnNext) btnNext.addEventListener('click', nextSlide);
  if (btnPrev) btnPrev.addEventListener('click', prevSlide);
  if (btnOverview) btnOverview.addEventListener('click', toggleOverview);
  if (closeOverviewBtn) closeOverviewBtn.addEventListener('click', closeOverview);
  if (btnFullscreen) btnFullscreen.addEventListener('click', toggleFullscreen);
  if (btnPrint) btnPrint.addEventListener('click', () => window.print());

  // Keyboard Shortcuts
  window.addEventListener('keydown', (e) => {
    // If overview is open, Escape closes it
    if (overviewModal && overviewModal.classList.contains('open')) {
      if (e.key === 'Escape') {
        closeOverview();
        return;
      }
    }

    switch (e.key) {
      case 'ArrowRight':
      case 'ArrowDown':
      case 'PageDown':
      case ' ': // Spacebar
        e.preventDefault();
        nextSlide();
        break;

      case 'ArrowLeft':
      case 'ArrowUp':
      case 'PageUp':
      case 'Backspace':
        e.preventDefault();
        prevSlide();
        break;

      case 'Home':
        e.preventDefault();
        goToSlide(0);
        break;

      case 'End':
        e.preventDefault();
        goToSlide(totalSlides - 1);
        break;

      case 'f':
      case 'F':
        toggleFullscreen();
        break;

      case 'o':
      case 'O':
      case 'Tab':
        e.preventDefault();
        toggleOverview();
        break;

      case 'p':
      case 'P':
        // Ctrl+P or plain P to trigger print
        if (!e.ctrlKey && !e.metaKey) {
          e.preventDefault();
          window.print();
        }
        break;

      case 'Escape':
        closeOverview();
        break;
    }
  });

  // Touch Swipe Support
  let touchStartX = 0;
  let touchEndX = 0;
  window.addEventListener('touchstart', (e) => {
    touchStartX = e.changedTouches[0].screenX;
  }, { passive: true });

  window.addEventListener('touchend', (e) => {
    touchEndX = e.changedTouches[0].screenX;
    const diff = touchEndX - touchStartX;
    if (Math.abs(diff) > 50) {
      if (diff < 0) nextSlide();
      else prevSlide();
    }
  }, { passive: true });

  // Read initial hash if present (#slide-3)
  const hash = window.location.hash;
  if (hash && hash.startsWith('#slide-')) {
    const slideNum = parseInt(hash.replace('#slide-', ''), 10);
    if (!isNaN(slideNum) && slideNum >= 1 && slideNum <= totalSlides) {
      goToSlide(slideNum - 1);
    } else {
      goToSlide(0);
    }
  } else {
    goToSlide(0);
  }
});
