use crossterm::{cursor, queue};
use std::io::{Stdout, Write};

/// Horizontal cell glyphs used to render partial bars.
const HBLOCKS: [char; 9] =
    [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'];

#[derive(Clone, Copy, PartialEq)]
pub enum Orient {
    Vertical,
    Horizontal,
}

/// Display mode: one bar per terminal row or dense averaging per row.
#[derive(Clone, Copy, PartialEq)]
pub enum Mode {
    Rows,
    Columns,
}

pub struct Layout {
    pub bars: usize, // analyzer resolution (bands)
    pub left_pad: u16,
    pub right_pad: u16,
    pub mode: Mode,
}

#[inline]
pub fn layout_for(
    w: u16,
    h: u16,
    _orient: Orient,
    mode: Mode,
) -> Layout {
    let left_pad = 1u16;
    let right_pad = 2u16;
    let usable_cols = w.saturating_sub(left_pad + right_pad);

    // Orientation no longer changes the drawing engine; we render horizontal bars.
    let bars = match mode {
        Mode::Rows => h.saturating_sub(3).max(1) as usize,
        Mode::Columns => usable_cols.max(10) as usize,
    };

    Layout {
        bars,
        left_pad,
        right_pad,
        mode,
    }
}

#[inline]
fn level_from_frac(frac: f32, gamma: f32) -> usize {
    let f = frac.clamp(0.0, 0.9999).powf(gamma);
    ((f * 8.0) + 0.5).floor().clamp(0.0, 8.0) as usize
}

#[inline]
fn h_partial(frac: f32) -> char {
    HBLOCKS[level_from_frac(frac, 0.70)]
}

/// Unified bar renderer for both orientations.
pub fn draw_bars(
    out: &mut Stdout,
    bars: &[f32],
    w: u16,
    h: u16,
    lay: &Layout,
) -> std::io::Result<()> {
    let rows = h.saturating_sub(3) as usize;
    let usable_w =
        w.saturating_sub(lay.left_pad + lay.right_pad) as usize;
    if rows == 0 || usable_w == 0 {
        return Ok(());
    }

    queue!(out, cursor::MoveTo(0, 1))?;

    let mut line = String::with_capacity(w as usize + 1);
    let bars_len = bars.len();

    match lay.mode {
        Mode::Rows => {
            // render available bars up to the number of rows
            let shown = bars_len.min(rows);
            for &v_raw in bars.iter().take(shown) {
                line.clear();
                line.extend(std::iter::repeat_n(
                    ' ',
                    lay.left_pad as usize,
                ));

                let v = v_raw.clamp(0.0, 1.0);
                let cells = v * usable_w as f32;
                let full = cells.floor() as usize;

                line.extend(std::iter::repeat_n(
                    '█',
                    full.min(usable_w),
                ));
                if full < usable_w {
                    let frac = (cells - full as f32).max(0.0);
                    line.push(if frac > 0.0 {
                        h_partial(frac)
                    } else {
                        ' '
                    });
                    if usable_w > full + 1 {
                        line.extend(std::iter::repeat_n(
                            ' ',
                            usable_w - full - 1,
                        ));
                    }
                }

                line.extend(std::iter::repeat_n(
                    ' ',
                    lay.right_pad as usize,
                ));
                line.push('\n');
                out.write_all(line.as_bytes())?;
            }

            // pad remaining rows with empties
            for _ in shown..rows {
                line.clear();
                line.extend(std::iter::repeat_n(
                    ' ',
                    lay.left_pad as usize,
                ));
                line.extend(std::iter::repeat_n(' ', usable_w));
                line.extend(std::iter::repeat_n(
                    ' ',
                    lay.right_pad as usize,
                ));
                line.push('\n');
                out.write_all(line.as_bytes())?;
            }
        }
        Mode::Columns => {
            // dense: many bands averaged into each row
            let per_row = bars_len.max(1).div_ceil(rows);
            for row in 0..rows {
                line.clear();
                line.extend(std::iter::repeat_n(
                    ' ',
                    lay.left_pad as usize,
                ));

                let start = row * per_row;
                if start < bars_len {
                    let end = (start + per_row).min(bars_len);
                    let mut acc = 0.0f32;
                    for &val in bars[start..end].iter() {
                        acc += val.clamp(0.0, 1.0);
                    }
                    let v =
                        (acc / (end - start) as f32).clamp(0.0, 1.0);

                    let cells = v * usable_w as f32;
                    let full = cells.floor() as usize;

                    line.extend(std::iter::repeat_n(
                        '█',
                        full.min(usable_w),
                    ));
                    if full < usable_w {
                        let frac = (cells - full as f32).max(0.0);
                        line.push(if frac > 0.0 {
                            h_partial(frac)
                        } else {
                            ' '
                        });
                        if usable_w > full + 1 {
                            line.extend(std::iter::repeat_n(
                                ' ',
                                usable_w - full - 1,
                            ));
                        }
                    }
                } else {
                    line.extend(std::iter::repeat_n(' ', usable_w));
                }

                line.extend(std::iter::repeat_n(
                    ' ',
                    lay.right_pad as usize,
                ));
                line.push('\n');
                out.write_all(line.as_bytes())?;
            }
        }
    }

    out.flush()?;
    Ok(())
}

// backwards-compat wrappers so the rest of the code can call either name
#[inline]
pub fn draw_blocks_horizontal(
    out: &mut Stdout,
    bars: &[f32],
    w: u16,
    h: u16,
    lay: &Layout,
    _mode: Mode,
) -> std::io::Result<()> {
    draw_bars(out, bars, w, h, lay)
}

#[inline]
pub fn draw_blocks_vertical(
    out: &mut Stdout,
    bars: &[f32],
    w: u16,
    h: u16,
    lay: &Layout,
) -> std::io::Result<()> {
    draw_bars(out, bars, w, h, lay)
}
