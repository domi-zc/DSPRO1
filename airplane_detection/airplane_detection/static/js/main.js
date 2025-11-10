// Scroll-Animation Airplane //
let ticking = false;
let scrollPositionPixel;
let scrollPositionPercentage;

let yValue;
let rotationValue;
let progress;
let progress_transform_y;
let progress_rotate;

let windowHeight = window.innerHeight;
let fullContentHeight = document.documentElement.scrollHeight;

// Activates when resizing window
window.addEventListener("resize", () => {
  windowHeight = window.innerHeight;
  fullContentHeight = document.documentElement.scrollHeight;
});

// Activates when scrolling
window.addEventListener("scroll", function (e) {
  scrollPositionPixel = window.pageYOffset;
  scrollPositionPercentage = Math.min(
    100,
    (scrollPositionPixel / (fullContentHeight - windowHeight)) * 100
  );

  // Airplane animation transformY() and rotate()
  if (scrollPositionPercentage <= 15) {
    yValue = -35;
    rotationValue = 0;
  } else if (scrollPositionPercentage <= 30) {
    progress = (scrollPositionPercentage - 15) / 15;
    yValue = -35 - (45 * progress);
    rotationValue = -20 * progress;
  } else if (scrollPositionPercentage <= 45) {
    progress = (scrollPositionPercentage - 30) / 15;
    yValue = -80; // Hält -80
    rotationValue = -20 + 20 * progress;
  } else if (scrollPositionPercentage <= 55) {
    yValue = -80; // Hält -80
    rotationValue = 0;
  } else if (scrollPositionPercentage <= 80) {
    progress_transform_y = (scrollPositionPercentage - 55) / 35;
    yValue = -80 + (45 * progress_transform_y);

    progress_rotate = (scrollPositionPercentage - 55) / 25;
    rotationValue = 20 * progress_rotate;
  } else if (scrollPositionPercentage <= 90) {
    progress_transform_y = (scrollPositionPercentage - 55) / 35;
    yValue = -80 + (45 * progress_transform_y);

    progress_rotate = (scrollPositionPercentage - 80) / 10;
    rotationValue = 20 - 20 * progress_rotate;
  } else {
    // 90-100
    yValue = -35;
    rotationValue = 0;
  }

  if (!ticking) {
    window.requestAnimationFrame(function () {
      document.getElementById("ap-header-logo").style.left =
        scrollPositionPercentage + "%";
      document.getElementById("ap-header-logo").style.transform =
        "translateX(-" +
        scrollPositionPercentage +
        "%) translateY(" +
        yValue +
        "%) rotate(" +
        rotationValue +
        "deg)";
      ticking = false;
    });
    ticking = true;
  }
});

// Aspect Ratio of Comparison-Slider Image //
window.addEventListener("load", () => {
  const cmp = document.getElementById("ap-compare-image");
  if (!cmp) return;

  const img = cmp.querySelector("img");
  if (!img) return;

  const w = img.naturalWidth;
  const h = img.naturalHeight;
  if (w > 0 && h > 0) {
    cmp.style.aspectRatio = `${w} / ${h}`;
  }
  windowHeight = window.innerHeight;
  fullContentHeight = document.documentElement.scrollHeight;
});

// Button "Start Process" Enablen und "ap-active" Klassen den Buttons zuweisen oder nehmen //
document.addEventListener("DOMContentLoaded", function () {
  const input = document.getElementById("ap-image-upload");
  const label = document.getElementById("ap-image-upload-label");
  const btn = document.getElementById("ap-process-btn");
  if (!input || !btn) return;

  function enableIfFileSelected() {
    const hasFile = !!(input.files && input.files.length > 0);
    if (hasFile && btn.disabled) {
      btn.disabled = false;
      btn.setAttribute("aria-disabled", "false");
      label.classList.remove("ap-active");
      btn.classList.add("ap-active");
    }
  }

  enableIfFileSelected();

  input.addEventListener("change", enableIfFileSelected);
  input.addEventListener("input", enableIfFileSelected);
});

// Image-Comparison Slider Slide Animation //
(function () {
  const cmp = document.getElementById("ap-compare-image");
  if (!cmp) return;

  function setPos(clientX) {
    const r = cmp.getBoundingClientRect();
    const pct = Math.min(
      100,
      Math.max(0, ((clientX - r.left) / r.width) * 100)
    );
    cmp.style.setProperty("--pos", pct + "%");
  }

  function onDown(e) {
    setPos(e.clientX);
    const move = (ev) => setPos(ev.clientX);
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }
  cmp.addEventListener("pointerdown", onDown);
})();

// Scroll-Up Button click //
initScrollUp();

function initScrollUp() {
  const el = document.getElementById("ap-scroll-up");
  if (!el) return;
  el.addEventListener("click", (e) => {
    e.preventDefault();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}
