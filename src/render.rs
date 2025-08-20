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
    Rows,     // one bar per terminal row
    Columns,  // many bars averaged into each row
}

#[derive(Clone, Copy, PartialEq)]
pub enum VertMode {
    Blocks,   // classic block renderer
    Braille,  // 2x4 subcell vertical resolution per column
}

pub struct Layout {
    pub bars: usize,
    pub left_pad: u16,
    pub right_pad: u16,
    pub horz_mode: HorzMode,
    pub vert_mode: VertMode,
}

#[inline]
pub fn layout_for(
    w: u16,
    h: u16,
    orient: Orient,
    hmode: HorzMode,
    vmode: VertMode,
) -> Layout {
    let left_pad = 1u16;
    let right_pad = 2u16;
    let usable_cols = w.saturating_sub(left_pad + right_pad);

    let bars = match orient {
        Orient::Horizontal => match hmode {
            HorzMode::Rows => h.saturating_sub(3).max(1) as usize,
            HorzMode::Columns => usable_cols.max(10) as usize,
        },
        Orient::Vertical => match vmode {
            // Braille uses 2 bars per character column
            VertMode::Braille => (usable_cols.max(1) as usize) * 2,
            VertMode::Blocks => usable_cols.max(10) as usize,
        },
    };

    Layout {
        bars,
        left_pad,
        right_pad,
        horz_mode: hmode,
        vert_mode: vmode,
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

// ---------------------- Vertical renderers ----------------------

pub fn draw_blocks_vertical(
    out: &mut Stdout,
    bars: &[f32],
    w: u16,
    h: u16,
    lay: &Layout,
) -> std::io::Result<()> {
    match lay.vert_mode {
        VertMode::Blocks => draw_vertical_blocks(out, bars, w, h, lay),
        VertMode::Braille => draw_vertical_braille(out, bars, w, h, lay),
    }
}

fn draw_vertical_blocks(
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
        for &v_in in bars.iter() {
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

/// Braille cell bit masks:
/// left column dots: 1,2,3,7  -> bits 0,1,2,6
/// right column dots: 4,5,6,8 -> bits 3,4,5,7
#[inline]
fn braille_mask_for_counts(left_sub: u8, right_sub: u8) -> u16 {
    let mut m: u16 = 0;
    match left_sub {
        0 => {}
        1 => { m |= 1 << 6; }                // dot 7
        2 => { m |= 1 << 6 | 1 << 2; }       // 7,3
        3 => { m |= 1 << 6 | 1 << 2 | 1 << 1; } // 7,3,2
        _ => { m |= 1 << 6 | 1 << 2 | 1 << 1 | 1 << 0; } // 7,3,2,1
    }
    match right_sub {
        0 => {}
        1 => { m |= 1 << 7; }                // dot 8
        2 => { m |= 1 << 7 | 1 << 5; }       // 8,6
        3 => { m |= 1 << 7 | 1 << 5 | 1 << 4; } // 8,6,5
        _ => { m |= 1 << 7 | 1 << 5 | 1 << 4 | 1 << 3; } // 8,6,5,4
    }
    m
}

fn draw_vertical_braille(
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
    let usable_w = w.saturating_sub(lay.left_pad + lay.right_pad) as usize;
    if usable_w == 0 {
        return Ok(());
    }

    // We expect bars.len() ~= usable_w * 2 (two bars per braille column).
    // If fewer, missing ones render as zero.
    queue!(out, cursor::MoveTo(0, 1))?;

    let mut line = String::with_capacity(w as usize + 1);
    for row_top in 0..rows {
        line.clear();
        let row_from_bottom = rows - 1 - row_top;

        // left padding
        line.extend(std::iter::repeat(' ').take(lay.left_pad as usize));

        for col in 0..usable_w {
            let li = col * 2;
            let ri = li + 1;

            let lv = bars.get(li).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            let rv = bars.get(ri).copied().unwrap_or(0.0).clamp(0.0, 1.0);

            let lcells = lv * rows as f32;
            let rcells = rv * rows as f32;

            let lsub = subrows_for_row(lcells, row_from_bottom);
            let rsub = subrows_for_row(rcells, row_from_bottom);

            let mask = braille_mask_for_counts(lsub, rsub);
            let ch = std::char::from_u32(0x2800 + mask as u32).unwrap_or(' ');
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

#[inline]
fn subrows_for_row(cells: f32, row_from_bottom: usize) -> u8 {
    // cells is total fill in row units. For this specific row, decide 0..4 subrows.
    let r0 = row_from_bottom as f32;
    let r1 = r0 + 1.0;

    if cells <= r0 + 1e-6 {
        0
    } else if cells >= r1 - 1e-6 {
        4
    } else {
        let frac = (cells - r0).clamp(0.0, 1.0);
        let sub = (frac * 4.0).ceil() as i32;
        sub.clamp(1, 4) as u8
    }
}

// ---------------------- Horizontal renderer ----------------------

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
            // classic look: one bar per row
            for row in 0..rows {
                line.clear();
                line.extend(std::iter::repeat(' ').take(lay.left_pad as usize));

                let v = if row < bars_len { bars[row] } else { 0.0 };
                let v = v.clamp(0.0, 1.0);
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

