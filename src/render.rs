use crossterm::{
    cursor, queue,
    style::{Color, ResetColor, SetBackgroundColor},
};
use std::io::{Stdout, Write};

// vertical partials from bottom: empty -> full
const VBLOCKS: [char; 9] =
    [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

// bar geometry
const BAR_W: usize = 2;
const GAP_W: usize = 1;

#[derive(Clone, Copy, PartialEq)]
pub enum Orient {
    Vertical,
    Horizontal,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Mode {
    Rows,
    Columns,
}

pub struct Layout {
    pub bars: usize, // analyzer resolution
    pub left_pad: u16,
    pub right_pad: u16,
    pub mode: Mode,
}

#[inline]
pub fn layout_for(
    w: u16,
    h: u16,
    orient: Orient,
    mode: Mode,
) -> Layout {
    let left_pad = 1u16;
    let right_pad = 2u16;
    let usable_cols = w.saturating_sub(left_pad + right_pad);

    let bars = match orient {
        Orient::Vertical => {
            let per = (BAR_W + GAP_W) as u16;
            ((usable_cols / per) as usize).max(1)
        }
        Orient::Horizontal => match mode {
            Mode::Rows => h.saturating_sub(3).max(1) as usize,
            Mode::Columns => usable_cols.max(10) as usize,
        },
    };

    Layout {
        bars,
        left_pad,
        right_pad,
        mode,
    }
}

// ---------------- HORIZONTAL (no partial glyphs) ----------------

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
            let shown = bars_len.min(rows);
            for &v_raw in bars.iter().take(shown) {
                line.clear();
                line.extend(std::iter::repeat_n(
                    ' ',
                    lay.left_pad as usize,
                ));

                let v = v_raw.clamp(0.0, 1.0);
                let full = (v * usable_w as f32).round() as usize;

                line.extend(std::iter::repeat_n(
                    '█',
                    full.min(usable_w),
                ));
                if usable_w > full {
                    line.extend(std::iter::repeat_n(
                        ' ',
                        usable_w - full,
                    ));
                }

                line.extend(std::iter::repeat_n(
                    ' ',
                    lay.right_pad as usize,
                ));
                line.push('\n');
                out.write_all(line.as_bytes())?;
            }

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
                    for &val in &bars[start..end] {
                        acc += val.clamp(0.0, 1.0);
                    }
                    let v =
                        (acc / (end - start) as f32).clamp(0.0, 1.0);
                    let full = (v * usable_w as f32).round() as usize;

                    line.extend(std::iter::repeat_n(
                        '█',
                        full.min(usable_w),
                    ));
                    if usable_w > full {
                        line.extend(std::iter::repeat_n(
                            ' ',
                            usable_w - full,
                        ));
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

// ---------------- VERTICAL with VBLOCKS ----------------

#[inline]
fn v_partial(frac: f32) -> char {
    let f = frac.clamp(0.0, 0.9999);
    let idx = ((f * 8.0) + 0.5).floor() as usize;
    VBLOCKS[idx.min(8)]
}

fn draw_columns_vertical(
    out: &mut Stdout,
    bars: &[f32],
    w: u16,
    h: u16,
    lay: &Layout,
) -> std::io::Result<()> {
    let rows = h.saturating_sub(3) as usize;
    let cols =
        w.saturating_sub(lay.left_pad + lay.right_pad) as usize;
    if rows == 0 || cols == 0 {
        return Ok(());
    }

    // visible bars from screen width
    let per = BAR_W + GAP_W;
    let visible = (cols / per).max(1);
    let n = bars.len().min(visible);

    // precompute full blocks and fractional top for each bar
    let mut fulls = vec![0usize; n];
    let mut fracs = vec![0f32; n];
    for i in 0..n {
        let height = bars[i].clamp(0.0, 1.0) * rows as f32;
        fulls[i] = height.floor() as usize;
        fracs[i] = height - fulls[i] as f32; // 0..1
    }

    // draw bottom to top, one terminal row each iteration
    for row in 0..rows {
        let y = h - 2 - row as u16;
        queue!(out, cursor::MoveTo(0, y))?;
        // left pad
        out.write_all(&vec![b' '; lay.left_pad as usize])?;

        for i in 0..n {
            let ch = if row < fulls[i] {
                '█'
            } else if row == fulls[i] && fracs[i] > 0.0 {
                v_partial(fracs[i])
            } else {
                ' '
            };

            // draw bar cells
            for _ in 0..BAR_W {
                out.write_all(
                    ch.encode_utf8(&mut [0; 4]).as_bytes(),
                )?;
            }
            // gap
            out.write_all(&vec![b' '; GAP_W])?;
        }

        // pad to end
        let used = n * per;
        if cols > used {
            out.write_all(&vec![b' '; cols - used])?;
        }
        out.write_all(&vec![b' '; lay.right_pad as usize])?;
    }

    // ensure no leftover background color
    queue!(out, ResetColor)?;
    out.flush()?;
    Ok(())
}

// wrappers

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
    draw_columns_vertical(out, bars, w, h, lay)
}
