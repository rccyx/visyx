use crossterm::{cursor, queue};
use std::cmp::Ordering;
use std::io::{Stdout, Write};

/// vertical partials (height)
const VBLOCKS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
/// horizontal partials (width)
const HBLOCKS: [char; 9] = [' ', '▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'];

#[derive(Clone, Copy, PartialEq)]
pub enum Orient {
    Vertical,
    Horizontal,
}

#[derive(Clone, Copy, PartialEq)]
pub enum HorzMode {
    Rows,     // one bar per terminal row (fills height)
    Columns,  // many bars averaged into each row (denser look)
}

pub struct Layout {
    pub bars: usize,
    pub left_pad: u16,
    pub right_pad: u16,
    pub horz_mode: HorzMode,
}

#[inline]
pub fn layout_for(w: u16, h: u16, orient: Orient, mode: HorzMode) -> Layout {
    let left_pad = 1u16;
    let right_pad = 2u16;
    let usable_cols = w.saturating_sub(left_pad + right_pad);
    let bars = match orient {
        Orient::Vertical => usable_cols.max(10) as usize,
        Orient::Horizontal => match mode {
            HorzMode::Rows => h.saturating_sub(3).max(1) as usize,
            HorzMode::Columns => usable_cols.max(10) as usize,
        },
    };
    Layout {
        bars,
        left_pad,
        right_pad,
        horz_mode: mode,
    }
}

#[inline]
fn level_from_frac(frac: f32, gamma: f32) -> usize {
    let f = frac.clamp(0.0, 0.9999).powf(gamma);
    ((f * 8.0) + 0.5).floor().clamp(0.0, 8.0) as usize
}

#[inline]
fn v_partial(frac: f32) -> char {
    VBLOCKS[level_from_frac(frac, 0.70)]
}

#[inline]
fn h_partial(frac: f32) -> char {
    HBLOCKS[level_from_frac(frac, 0.70)]
}

pub fn draw_blocks_vertical(
    out: &mut Stdout,
    bars: &[f32],
    w: u16,
    h: u16,
    lay: &Layout,
) -> std::io::Result<()> {
    let rows = h.saturating_sub(3) as usize;
    if rows == 0 {
        return Ok(());
    }

    queue!(out, cursor::MoveTo(0, 1))?;

    let mut line = String::with_capacity(w as usize + 1);
    for row_top in 0..rows {
        line.clear();
        let row_from_bottom = rows - 1 - row_top;

        // left padding
        line.extend(std::iter::repeat(' ').take(lay.left_pad as usize));

        // columns across the width
        for (i, &v_in) in bars.iter().enumerate() {
            let v = v_in.clamp(0.0, 1.0);
            let cells = v * rows as f32;
            let full = cells.floor() as usize;

            let ch = match row_from_bottom.cmp(&full) {
                Ordering::Less => '█',
                Ordering::Equal => {
                    let frac = (cells - full as f32).max(0.0);
                    if frac > 0.0 { v_partial(frac) } else { ' ' }
                }
                Ordering::Greater => ' ',
            };
            line.push(ch);
        }

        // right padding
        line.extend(std::iter::repeat(' ').take(lay.right_pad as usize));

        line.push('\n');
        out.write_all(line.as_bytes())?;
    }

    out.flush()?;
    Ok(())
}

pub fn draw_blocks_horizontal(
    out: &mut Stdout,
    bars: &[f32],
    w: u16,
    h: u16,
    lay: &Layout,
    mode: HorzMode,
) -> std::io::Result<()> {
    let rows = h.saturating_sub(3) as usize;
    let usable_w = w.saturating_sub(lay.left_pad + lay.right_pad) as usize;
    if rows == 0 || usable_w == 0 {
        return Ok(());
    }

    queue!(out, cursor::MoveTo(0, 1))?;

    let mut line = String::with_capacity(w as usize + 1);
    let bars_len = bars.len();

    match mode {
        HorzMode::Rows => {
            // classic look: one bar per row, fills the height
            for row in 0..rows {
                line.clear();
                line.extend(std::iter::repeat(' ').take(lay.left_pad as usize));

                let v = if row < bars_len { bars[row] } else { 0.0 };
                let v = v.clamp(0.0, 1.0);
                let cells = v * usable_w as f32;
                let full = cells.floor() as usize;

                // full blocks
                line.extend(std::iter::repeat('█').take(full.min(usable_w)));
                // partial
                if full < usable_w {
                    let frac = (cells - full as f32).max(0.0);
                    line.push(if frac > 0.0 { h_partial(frac) } else { ' ' });
                    if usable_w > full + 1 {
                        line.extend(std::iter::repeat(' ').take(usable_w - full - 1));
                    }
                }

                line.extend(std::iter::repeat(' ').take(lay.right_pad as usize));
                line.push('\n');
                out.write_all(line.as_bytes())?;
            }
        }
        HorzMode::Columns => {
            // dense look: many bars averaged into each row
            let per_row = (bars_len.max(1) + rows - 1) / rows; // ceil
            for row in 0..rows {
                line.clear();
                line.extend(std::iter::repeat(' ').take(lay.left_pad as usize));

                let start = row * per_row;
                if start < bars_len {
                    let end = (start + per_row).min(bars_len);
                    let mut acc = 0.0f32;
                    for i in start..end {
                        acc += bars[i].clamp(0.0, 1.0);
                    }
                    let v = (acc / (end - start) as f32).clamp(0.0, 1.0);

                    let cells = v * usable_w as f32;
                    let full = cells.floor() as usize;

                    line.extend(std::iter::repeat('█').take(full.min(usable_w)));
                    if full < usable_w {
                        let frac = (cells - full as f32).max(0.0);
                        line.push(if frac > 0.0 { h_partial(frac) } else { ' ' });
                        if usable_w > full + 1 {
                            line.extend(std::iter::repeat(' ').take(usable_w - full - 1));
                        }
                    }
                } else {
                    line.extend(std::iter::repeat(' ').take(usable_w));
                }

                line.extend(std::iter::repeat(' ').take(lay.right_pad as usize));
                line.push('\n');
                out.write_all(line.as_bytes())?;
            }
        }
    }

    out.flush()?;
    Ok(())
}

