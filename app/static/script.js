/* ---------- helpers ---------- */
const $ = id => document.getElementById(id);
const postImg = async dataURL =>
  (await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({image:dataURL})})).json();

/* ---------- canvas setup ---------- */
const cvs = $("draw");
const ctx = cvs.getContext("2d");
ctx.fillStyle = "#ffffff"; ctx.fillRect(0,0,280,280);

let drawing = false, lastX=0, lastY=0;
const line = (x1,y1,x2,y2)=>{
  ctx.strokeStyle="#000"; ctx.lineWidth=18; ctx.lineCap="round";
  ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke();
};

/* mouse events */
cvs.addEventListener("mousedown",e=>{drawing=true;[lastX,lastY]=[e.offsetX,e.offsetY];});
cvs.addEventListener("mousemove",e=>{if(!drawing)return; line(lastX,lastY,e.offsetX,e.offsetY); [lastX,lastY]=[e.offsetX,e.offsetY];});
["mouseup","mouseleave"].forEach(ev=>cvs.addEventListener(ev,()=>drawing=false));

/* touch events for smooth drawing */
cvs.addEventListener("touchstart",e=>{
  e.preventDefault(); drawing=true;
  const rect=cvs.getBoundingClientRect();
  [lastX,lastY]=[e.touches[0].clientX-rect.left, e.touches[0].clientY-rect.top];
});
cvs.addEventListener("touchmove",e=>{
  e.preventDefault(); if(!drawing)return;
  const rect=cvs.getBoundingClientRect();
  const x=e.touches[0].clientX-rect.left, y=e.touches[0].clientY-rect.top;
  line(lastX,lastY,x,y); [lastX,lastY]=[x,y];
});
cvs.addEventListener("touchend",()=>drawing=false);

/* ---------- canvas buttons ---------- */
$("clearBtn").onclick = () => { ctx.fillStyle="#fff"; ctx.fillRect(0,0,280,280); $("canvasResult").innerText=""; };

$("predictCanvasBtn").onclick = async () => {
  $("canvasResult").innerText = "Predicting…";
  const res = await postImg(cvs.toDataURL("image/png"));
  $("canvasResult").innerText = res.pred ? `${res.pred} (${res.conf})` : "Error";
};

/* ---------- upload flow ---------- */
const fileInput = $("fileInput");
$("uploadBtn").onclick = ()=> fileInput.click();

fileInput.onchange = e => {
  const file = e.target.files[0];
  if(!file) return;
  const reader = new FileReader();
  reader.onload = async ev => {
    const dataURL = ev.target.result;
    $("preview").src = dataURL; $("preview").style.display="block";
    $("uploadResult").innerText = "Predicting…";
    const res = await postImg(dataURL);
    $("uploadResult").innerText = res.pred ? `${res.pred} (${res.conf})` : "Error";
  };
  reader.readAsDataURL(file);
};
