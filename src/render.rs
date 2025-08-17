use crossterm::{cursor, queue};
use std::cmp::Ordering;
use std::io::{Stdout, Write};

const BLOCKS: [char; 9] =
    [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
const BAYER8: [[u8; 8]; 8] = [
    [0, 48, 12, 60, 3, 51, 15, 63],
    [32, 16, 44, 28, 35, 19, 47, 31],
    [8, 56, 4, 52, 11, 59, 7, 55],
    [40, 24, 36, 20, 43, 27, 39, 23],
    [2, 50, 14, 62, 1, 49, 13, 61],
    [34, 18, 46, 30, 33, 17, 45, 29],
    [10, 58, 6, 54, 9, 57, 5, 53],
    [42, 26, 38, 22, 41, 25, 37, 21],
];

#[derive(Clone, Copy, PartialEq)]
pub enum Orient {
    Vertical,
    Horizontal,
}

pub struct Layout {
    pub bars: usize,
    pub left_pad: u16,
    pub right_pad: u16,
}

pub fn layout_for(w: u16, _h: u16, orient: Orient) -> Layout {
    match orient {
        Orient::Vertical => {
            let left_pad = 1u16;
            let right_pad = 2u16;
            let usable = w.saturating_sub(left_pad + right_pad);
            Layout {
                bars: usable.max(10) as usize,
                left_pad,
                right_pad,
            }
        }
        Orient::Horizontal => Layout {
            bars: 0,
            left_pad: 1,
            right_pad: 2,
        },
    }
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
    for row_top in 0..rows {
        let row_from_bottom = rows - 1 - row_top;
        let mut line = String::with_capacity(w as usize);
        for _ in 0..lay.left_pad {
            line.push(' ');
        }
        for (i, &v) in bars.iter().enumerate() {
            let v = v.clamp(0.0, 1.0);
            let cells = v * rows as f32;
            let full = cells.floor() as usize;
            let frac = (cells - full as f32).clamp(0.0, 0.999_9);

            let ch = match row_from_bottom.cmp(&full) {
                Ordering::Less => '█',
                Ordering::Equal => {
                    let threshold =
                        BAYER8[row_top & 7][i & 7] as f32 / 64.0;
                    let mut level = (frac * 8.0).floor();
                    if frac.fract() > threshold {
                        level += 1.0;
                    }
                    BLOCKS[level.clamp(0.0, 8.0) as usize]
                }
                Ordering::Greater => ' ',
            };
            line.push(ch);
        }
        for _ in 0..lay.right_pad {
            line.push(' ');
        }
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
) -> std::io::Result<()> {
    let rows = h.saturating_sub(3) as usize;
    let usable_w =
        w.saturating_sub(lay.left_pad + lay.right_pad) as usize;
    if rows == 0 || usable_w == 0 {
        return Ok(());
    }

    queue!(out, cursor::MoveTo(0, 1))?;
    for row in 0..rows.min(bars.len()) {
        let v = bars[row].clamp(0.0, 1.0);
        let cells = v * usable_w as f32;
        let full = cells.floor() as usize;
        let frac = (cells - full as f32).clamp(0.0, 0.999_9);

        let mut line = String::with_capacity(w as usize);
        for _ in 0..lay.left_pad {
            line.push(' ');
        }
        for _ in 0..full {
            line.push('█');
        }
        if full < usable_w {
            let threshold = BAYER8[row & 7][full & 7] as f32 / 64.0;
            let mut level = (frac * 8.0).floor();
            if frac.fract() > threshold {
                level += 1.0;
            }
            line.push(BLOCKS[level.clamp(0.0, 8.0) as usize]);
        }
        while line.chars().count()
            < (lay.left_pad as usize + usable_w)
        {
            line.push(' ');
        }
        for _ in 0..lay.right_pad {
            line.push(' ');
        }
        line.push('\n');
        out.write_all(line.as_bytes())?;
    }
    for _ in bars.len()..rows {
        let mut line = String::new();
        for _ in 0..w {
            line.push(' ');
        }
        line.push('\n');
        out.write_all(line.as_bytes())?;
    }
    out.flush()?;
    Ok(())
}
