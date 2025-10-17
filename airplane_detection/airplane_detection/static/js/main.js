// Scroll-Animation Airplane //
let ticking = false;
let scrollPositionPixel;
let scrollPositionPercentage;

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

  if (!ticking) {
    window.requestAnimationFrame(function () {
      document.getElementById("ap-header-logo").style.left =
        scrollPositionPercentage + "%";
      document.getElementById("ap-header-logo").style.transform =
        "translateX(-" + scrollPositionPercentage + "%) translateY(-50%)";
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
  input.addEventListener("input",  enableIfFileSelected);
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