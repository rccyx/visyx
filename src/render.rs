use crossterm::{cursor, queue};
use std::io::{Stdout, Write};

// vertical partials from bottom: empty -> full
const VBLOCKS: [char; 9] =
    [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

// bar geometry
const BAR_W: usize = 2;
const GAP_W: usize = 1;

pub struct Layout {
    pub bars: usize, // analyzer resolution (bands)
    pub left_pad: u16,
    pub right_pad: u16,
    pub top_pad: u16, // rows reserved at top (0 when hud off)
}

#[inline]
pub fn layout_for(w: u16, top_pad: u16) -> Layout {
    let left_pad = 1u16;
    let right_pad = 2u16;
    let usable_cols = w.saturating_sub(left_pad + right_pad);

    // one bar per (BAR_W + GAP_W) columns
    let per = (BAR_W + GAP_W) as u16;
    let bars = ((usable_cols / per) as usize).max(1);

    Layout {
        bars,
        left_pad,
        right_pad,
        top_pad,
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
