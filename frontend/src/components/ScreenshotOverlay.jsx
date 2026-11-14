import { categoryStyle } from '../format.js';

export default function ScreenshotOverlay({ imageUrl, imageSize, findings, activeId, onSelect }) {
  const W = imageSize?.width;
  const H = imageSize?.height;
  const canPlace = W > 0 && H > 0;
  const boxes = findings.filter((f) => f.source_location?.bbox);

  return (
    <figure>
      <div className="relative w-full overflow-hidden rounded-lg border border-slate-200 bg-slate-100">
        <img
          src={imageUrl}
          alt="Uploaded screenshot with classified text regions outlined"
          className="block h-auto w-full"
        />
        {canPlace &&
          boxes.map((f) => {
            const b = f.source_location.bbox;
            const style = categoryStyle(f.category);
            const active = activeId === f.id;
            return (
              <button
                key={f.id}
                type="button"
                onClick={() => onSelect(f.id)}
                aria-label={`${f.category_label} region, jump to finding`}
                className={`absolute overflow-hidden border-2 ${style.box} ${
                  active ? 'ring-2 ring-teal-600 ring-offset-1' : ''
                }`}
                style={{
                  left: `${(b.x / W) * 100}%`,
                  top: `${(b.y / H) * 100}%`,
                  width: `${(b.width / W) * 100}%`,
                  height: `${(b.height / H) * 100}%`,
                }}
              >
                <span
                  className={`absolute left-0 top-0 max-w-full truncate px-1 py-px text-[10px] font-semibold leading-tight text-white ${style.tag}`}
                >
                  {f.category_label}
                </span>
              </button>
            );
          })}
      </div>
      {!canPlace && (
        <p className="mt-2 text-xs text-amber-800">Image dimensions were not returned, so the boxes cannot be placed.</p>
      )}
      <figcaption className="mt-2 text-xs leading-relaxed text-slate-500">
        Boxes mark OCR text regions classified by the text model — this is not a visual-layout classifier.
      </figcaption>
    </figure>
  );
}
