; Color grading of source.png in GIMP (batch mode), used to produce the graded.png files.
; Only global per-pixel operations: curves, color balance, hue/saturation.

; Run from this folder (all calls in one -b: separate -b expressions do not share definitions):
;   gimp -i -b "(begin (load \"grade_gimp.scm\") (grade-look \"film_warm\"))" -b "(gimp-quit 0)"
; Values for curves are in [0, 1].

(define (load-source)
  (let* ((image (car (gimp-file-load RUN-NONINTERACTIVE "source.png" "source.png")))
         (drawable (car (gimp-image-get-active-layer image))))
    (list image drawable)))

(define (save-graded image drawable dir)
  (let ((path (string-append dir "/graded.png")))
    (gimp-file-save RUN-NONINTERACTIVE image drawable path path)
    (gimp-image-delete image)))

(define (curve drawable channel points)
  (gimp-drawable-curves-spline drawable channel (length points) (list->vector points)))

(define (grade-look name)
  (let* ((loaded (load-source))
         (image (car loaded))
         (d (cadr loaded)))
    (cond
      ; Warm filmic look: gentle S-curve, slightly lifted blacks, warm mids and highlights, +sat
      ((string=? name "film_warm")
        (curve d HISTOGRAM-VALUE '(0.0 0.03  0.25 0.22  0.5 0.53  0.75 0.80  1.0 0.98))
        (gimp-drawable-color-balance d TRANSFER-SHADOWS TRUE -2 0 3)
        (gimp-drawable-color-balance d TRANSFER-MIDTONES TRUE 9 1 -12)
        (gimp-drawable-color-balance d TRANSFER-HIGHLIGHTS TRUE 10 2 -12)
        (gimp-drawable-hue-saturation d HUE-RANGE-ALL 0 0 15 0))
      ; Teal & orange blockbuster look: teal shadows, warm highlights, more contrast
      ((string=? name "teal_orange")
        (curve d HISTOGRAM-VALUE '(0.0 0.0  0.22 0.17  0.5 0.5  0.78 0.84  1.0 1.0))
        (gimp-drawable-color-balance d TRANSFER-SHADOWS TRUE -12 0 8)
        (gimp-drawable-color-balance d TRANSFER-MIDTONES TRUE 10 0 -10)
        (gimp-drawable-color-balance d TRANSFER-HIGHLIGHTS TRUE 14 0 -16)
        (gimp-drawable-hue-saturation d HUE-RANGE-ALL 0 0 10 0))
      ; Cold "bleach bypass": desaturated, high contrast, blue-green tint
      ((string=? name "cold_bleach")
        (gimp-drawable-hue-saturation d HUE-RANGE-ALL 0 0 -45 0)
        (curve d HISTOGRAM-VALUE '(0.0 0.0  0.25 0.17  0.5 0.5  0.75 0.86  1.0 1.0))
        (curve d HISTOGRAM-BLUE '(0.0 0.04  0.5 0.54  1.0 1.0))
        (gimp-drawable-color-balance d TRANSFER-MIDTONES TRUE -8 4 10)))
    (save-graded image d name)))
