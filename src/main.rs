use image::io::Reader as ImageReader;
use image::{imageops, DynamicImage, GenericImageView, ImageOutputFormat, RgbImage};
use minifb::{Key, Window, WindowOptions};
use ndarray::{s, Array, Array1, Array2, Array4, ArrayBase, Axis, Data, Ix2, OwnedRepr, Zip};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::{
    utils::{frame_formats, CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};
use ort::{self, inputs, CPUExecutionProvider, CUDAExecutionProvider, GraphOptimizationLevel};
use ort::{Session, TensorElementType, ValueType};
use std::error::Error;

const WIDTH: usize = 320;
const HEIGHT: usize = 240;
fn display_element_type(t: TensorElementType) -> &'static str {
    match t {
        TensorElementType::Bfloat16 => "bf16",
        TensorElementType::Bool => "bool",
        TensorElementType::Float16 => "f16",
        TensorElementType::Float32 => "f32",
        TensorElementType::Float64 => "f64",
        TensorElementType::Int16 => "i16",
        TensorElementType::Int32 => "i32",
        TensorElementType::Int64 => "i64",
        TensorElementType::Int8 => "i8",
        TensorElementType::String => "str",
        TensorElementType::Uint16 => "u16",
        TensorElementType::Uint32 => "u32",
        TensorElementType::Uint64 => "u64",
        TensorElementType::Uint8 => "u8",
    }
}

fn display_value_type(value: &ValueType) -> String {
    match value {
        ValueType::Tensor { ty, dimensions } => {
            format!(
                "Tensor<{}>({})",
                display_element_type(*ty),
                dimensions
                    .iter()
                    .map(|c| if *c == -1 {
                        "dyn".to_string()
                    } else {
                        c.to_string()
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        ValueType::Map { key, value } => format!(
            "Map<{}, {}>",
            display_element_type(*key),
            display_element_type(*value)
        ),
        ValueType::Sequence(inner) => format!("Sequence<{}>", display_value_type(inner)),
    }
}
fn create_window(title: &str, width: usize, height: usize) -> Result<Window, minifb::Error> {
    Window::new(title, width, height, WindowOptions::default())
}

fn initialize_camera() -> Result<Camera, nokhwa::NokhwaError> {
    let ff = frame_formats();
    let fmt = RequestedFormat::with_formats(RequestedFormatType::AbsoluteHighestFrameRate, ff);
    let mut cam = Camera::new(CameraIndex::Index(0), fmt)?;
    cam.open_stream()?;
    Ok(cam)
}

fn preprocess(img: &RgbImage) -> ndarray::Array4<f32> {
    let resized = image::imageops::resize(img, 320, 240, image::imageops::FilterType::Nearest);
    let (width, height) = resized.dimensions();
    let raw_data = resized.into_raw(); // This gives us Vec<u8> of the data
    let arr_data = Array::from_shape_vec((height as usize, width as usize, 3), raw_data)
        .expect("Error converting raw buffer to ndarray");
    let normalized = arr_data.mapv(|x| (x as f32 - 127.0) / 128.0);
    let transposed = normalized.permuted_axes([2, 0, 1]);
    let expanded = transposed.insert_axis(Axis(0));

    expanded
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut window = create_window("Raqote", WIDTH, HEIGHT)?;
    let mut cam = initialize_camera()?;
    // let input = preprocess(original_img.clone().as_mut_rgb8().unwrap());
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(r"C:\Users\anand\.rust\webcamcapture\version-RFB-320.onnx")?;

    println!("Inputs:");
    for (i, input) in session.inputs.iter().enumerate() {
        println!(
            "    {i} {}: {}",
            input.name,
            display_value_type(&input.input_type)
        );
    }
    println!("Outputs:");
    for (i, output) in session.outputs.iter().enumerate() {
        println!(
            "    {i} {}: {}",
            output.name,
            display_value_type(&output.output_type)
        );
    }
    while window.is_open() && !window.is_key_down(Key::Escape) {
        if let Ok(frame) = cam.frame() {
            let img = frame.decode_image::<RgbFormat>().unwrap(); // Assumes successful decoding
                                                                  // Resize the image
            let rimg = imageops::resize(&img, WIDTH as u32, HEIGHT as u32, imageops::Lanczos3);

            let buffer: Vec<u32> = rimg
                .enumerate_pixels()
                .map(|(_, _, pixel)| {
                    let [r, g, b] = pixel.0;
                    ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
                })
                .collect();

            let input = preprocess(&img);
            let outputs = session.run(inputs!["input" => input.view()]?)?;
            let scores = outputs["scores"]
                .try_extract_tensor::<f32>()?
                .t()
                .into_owned();
            let boxes = outputs["boxes"]
                .try_extract_tensor::<f32>()?
                .t()
                .into_owned();
            let boxes = boxes.permuted_axes(vec![2, 1, 0]);
            println!("Scores: {:?}", scores);
            println!("Boxes: {:?}", boxes.shape());
            window.update_with_buffer(&buffer, WIDTH, HEIGHT)?;
        }
    }

    cam.stop_stream()?;
    Ok(())
}
