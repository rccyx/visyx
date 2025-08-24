use crossterm::{cursor, queue};
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
    pub bars: usize, // analyzer resolution (bands)
    pub left_pad: u16,
    pub right_pad: u16,
    pub top_pad: u16, // rows reserved at top (0 when hud off)
    pub mode: Mode,
}

#[inline]
pub fn layout_for(
    w: u16,
    h: u16,
    orient: Orient,
    mode: Mode,
    top_pad: u16,
) -> Layout {
    let left_pad = 1u16;
    let right_pad = 2u16;
    let usable_cols = w.saturating_sub(left_pad + right_pad);

    let bars = match orient {
        // one bar per (BAR_W + GAP_W) columns
        Orient::Vertical => {
            let per = (BAR_W + GAP_W) as u16;
            ((usable_cols / per) as usize).max(1)
        }
        Orient::Horizontal => match mode {
            // one bar per remaining row
            Mode::Rows => h.saturating_sub(top_pad).max(1) as usize,
            // one bar per column
            Mode::Columns => usable_cols.max(10) as usize,
        },
    };

    Layout {
        bars,
        left_pad,
        right_pad,
        top_pad,
        mode,
    }
}

#[inline]
fn v_partial(frac: f32) -> char {
    let f = frac.clamp(0.0, 0.9999);
    let idx = ((f * 8.0) + 0.5).floor() as usize;
    VBLOCKS[idx.min(8)]
}

// write n spaces without allocating a vec
#[inline]
fn write_spaces(
    out: &mut Stdout,
    mut n: usize,
) -> std::io::Result<()> {
    const BLANK: [u8; 64] = [b' '; 64];
    while n >= BLANK.len() {
        out.write_all(&BLANK)?;
        n -= BLANK.len();
    }
    if n > 0 {
        out.write_all(&BLANK[..n])?;
    }
    Ok(())
}

// horizontal

pub fn draw_bars(
    out: &mut Stdout,
    bars: &[f32],
    w: u16,
    h: u16,
    lay: &Layout,
) -> std::io::Result<()> {
    let rows = h.saturating_sub(lay.top_pad) as usize;
    let usable_w =
        w.saturating_sub(lay.left_pad + lay.right_pad) as usize;
    if rows == 0 || usable_w == 0 {
        return Ok(());
    }

    queue!(out, cursor::MoveTo(0, lay.top_pad))?;

    let mut line = String::with_capacity((w as usize + 1).max(256));
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

// vertical with vblocks

fn draw_columns_vertical(
    out: &mut Stdout,
    bars: &[f32],
    w: u16,
    h: u16,
    lay: &Layout,
) -> std::io::Result<()> {
    let rows = h.saturating_sub(lay.top_pad) as usize;
    let cols =
        w.saturating_sub(lay.left_pad + lay.right_pad) as usize;
    if rows == 0 || cols == 0 {
        return Ok(());
    }

    let per = BAR_W + GAP_W;
    let n = bars.len().min((cols / per).max(1));

    // precompute heights
    let mut fulls = vec![0usize; n];
    let mut fracs = vec![0f32; n];
    for i in 0..n {
        let height = bars[i].clamp(0.0, 1.0) * rows as f32;
        fulls[i] = height.floor() as usize;
        fracs[i] = height - fulls[i] as f32;
    }

    // bottom to top
    for row in 0..rows {
        let y = h - 1 - row as u16;
        queue!(out, cursor::MoveTo(0, y))?;
        write_spaces(out, lay.left_pad as usize)?;

        for i in 0..n {
            let ch = if row < fulls[i] {
                '█'
            } else if row == fulls[i] && fracs[i] > 0.0 {
                v_partial(fracs[i])
            } else {
                ' '
            };

            for _ in 0..BAR_W {
                out.write_all(
                    ch.encode_utf8(&mut [0; 4]).as_bytes(),
                )?;
            }
            write_spaces(out, GAP_W)?;
        }

        let used = n * per;
        if cols > used {
            write_spaces(out, cols - used)?;
        }
        write_spaces(out, lay.right_pad as usize)?;
    }

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
