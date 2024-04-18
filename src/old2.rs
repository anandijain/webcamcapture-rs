use font_kit::family_name::FamilyName;
use font_kit::properties::Properties;
use font_kit::source::SystemSource;
use minifb::{Key, MouseMode, Scale, ScaleMode, Window, WindowOptions};
use raqote::{
    DrawOptions, DrawTarget, PathBuilder, Point, SolidSource, Source, StrokeStyle, Transform,
};
const WIDTH: usize = 400;
const HEIGHT: usize = 400;
// use image::{io::image as ImageReader};
fn main() {
    let mut window = Window::new(
        "Raqote",
        WIDTH,
        HEIGHT,
        WindowOptions {
            ..WindowOptions::default()
        },
    )
    .unwrap();
    
    let img = image::open("me.jpeg").unwrap();
    
    // Resize the image to fit the window
    let img = img.resize_exact(WIDTH as u32, HEIGHT as u32, image::imageops::Lanczos3);

    // Convert the image to a vector of u32 RGBA values
    let buffer: Vec<u32> = img
        .to_rgba8()
        .chunks(4)
        .map(|c| {
            ((c[3] as u32) << 24) | ((c[0] as u32) << 16) | ((c[1] as u32) << 8) | (c[2] as u32)
        })
        .collect();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Update the window with the image buffer
        window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();
    }
    loop {
        dt.clear(SolidSource::from_unpremultiplied_argb(
            0xff, 0xff, 0xff, 0xff,
        ));
        let mut pb = PathBuilder::new();
        if let Some(pos) = window.get_mouse_pos(MouseMode::Clamp) {
            pb.rect(pos.0, pos.1, 100., 130.);
            let path = pb.finish();
            dt.fill(
                &path,
                &Source::Solid(SolidSource::from_unpremultiplied_argb(0xff, 0, 0xff, 0)),
                &DrawOptions::new(),
            );

            let pos_string = format!("{:?}", pos);
            dt.draw_text(
                &font,
                36.,
                &pos_string,
                Point::new(0., 100.),
                &Source::Solid(SolidSource::from_unpremultiplied_argb(0xff, 0, 0, 0)),
                &DrawOptions::new(),
            );

            window
                .update_with_buffer(dt.get_data(), size.0, size.1)
                .unwrap();
        }
    }
}
