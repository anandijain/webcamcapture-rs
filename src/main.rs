use image::imageops;
use image::io::Reader as ImageReader;
use image::ImageBuffer;
use minifb::{Key, MouseMode, Scale, ScaleMode, Window, WindowOptions};
use nokhwa::{
    pixel_format::RgbFormat,
    utils::{frame_formats, CameraIndex, RequestedFormat, RequestedFormatType},
    Camera, FormatDecoder,
};
use raqote::{
    DrawOptions, DrawTarget, PathBuilder, Point, SolidSource, Source, StrokeStyle, Transform,
};

const WIDTH: usize = 400;
const HEIGHT: usize = 400;

fn main() -> Result<(), anyhow::Error> {
    let mut window = Window::new(
        "Raqote",
        WIDTH,
        HEIGHT,
        WindowOptions {
            ..WindowOptions::default()
        },
    )
    .unwrap();

    let ff = frame_formats();
    let fmt = RequestedFormat::with_formats(RequestedFormatType::None, ff);
    let mut cam = Camera::new(CameraIndex::Index(0), fmt).unwrap();
    println!("Camera opened: {:?}", cam.camera_format());
    let x = cam.open_stream();
    println!("Stream opened: {:?}", x);
    let f = cam.frame().unwrap();
    let img = ImageReader::open("me.jpeg")?.decode()?;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        // Update the window with the image buffer
        let f = cam.frame().unwrap();
        let img = f.decode_image::<RgbFormat>().unwrap();
        let rimg = imageops::resize(&img, WIDTH as u32, HEIGHT as u32, imageops::Lanczos3);
        let buffer: Vec<u32> = rimg
            // .to_rgb8()
            .chunks(3)
            .map(|c| {
                ((c[0] as u32) << 16) | ((c[1] as u32) << 8) | (c[2] as u32)
            })
            .collect();

        window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();
        // let img = img.resize(WIDTH as u32, HEIGHT as u32, image::imageops::Lanczos3);
    }
    cam.stop_stream().unwrap();
    // println!("DecodedFrame of {}", decoded.len());
    // decoded.save("foo.jpeg").unwrap();
    Ok(())
}
