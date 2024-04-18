use std::collections::BinaryHeap;
use std::time::Instant;

use image::imageops::FilterType;
use image::{GenericImageView, RgbImage};
use ndarray::{s, Array, Array1, Array2, Array4, ArrayBase, Axis, Data, Ix2, OwnedRepr, Zip};
use ndarray_npy;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::{
    utils::{frame_formats, CameraIndex, RequestedFormat, RequestedFormatType},
    Camera, FormatDecoder,
};
use ort::{self, inputs, CPUExecutionProvider, CUDAExecutionProvider, GraphOptimizationLevel};
use ort::{Session, TensorElementType, ValueType};
use show_image::{create_window, ImageView};
use std::ops::Index;
use anyhow;
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

fn take_pic() {
    let ff = frame_formats();
    let fmt = RequestedFormat::with_formats(RequestedFormatType::None, ff);
    let mut cam = Camera::new(CameraIndex::Index(0), fmt).unwrap();
    // println!("Camera opened: {:?}", cam);
    println!("Camera opened: {:?}", cam.camera_format());
    let x = cam.open_stream();
    println!("Stream opened: {:?}", x);
    let f = cam.frame().unwrap();
    let d = RgbFormat::write_output(f.source_frame_format(), f.resolution(), f.buffer()).unwrap();

    // let mut camera = Camera::new(index, requested).unwrap();
    // camera.open_stream().unwrap();
    // let frame = camera.frame().unwrap();
    cam.stop_stream().unwrap();
    println!("Captured Single Frame of {}", f.buffer().len());
    let decoded = f.decode_image::<RgbFormat>().unwrap();
    println!("DecodedFrame of {}", decoded.len());
    decoded.save("foo.jpeg").unwrap();
}

fn preprocess(img: &RgbImage) -> ndarray::Array4<f32> {
    // Convert to RGB
    // let rgb_img = img.to_rgb8();

    // Resize image using nearest neighbor algorithm
    let resized = image::imageops::resize(img, 320, 240, image::imageops::FilterType::Nearest);

    // Create an ndarray from the image data
    let (width, height) = resized.dimensions();
    let raw_data = resized.into_raw(); // This gives us Vec<u8> of the data
    let arr_data = Array::from_shape_vec((height as usize, width as usize, 3), raw_data)
        .expect("Error converting raw buffer to ndarray");

    // Normalize the image data
    let normalized = arr_data.mapv(|x| (x as f32 - 127.0) / 128.0);

    // Transpose axis from (240, 320, 3) to (3, 240, 320)
    let transposed = normalized.permuted_axes([2, 0, 1]);

    // Expand dimensions to add batch axis
    let expanded = transposed.insert_axis(Axis(0));

    expanded
}

// fn area_of<S1, S2>(left_top: &ArrayBase<S1, Ix2>, right_bottom: &ArrayBase<S2, Ix2>) -> Array1<f32>
// where
//     S1: Data<Elem = f32>,
//     S2: Data<Elem = f32>,
// {
//     let mut areas = Array1::<f32>::zeros(left_top.dim().0);
//     Zip::from(&mut areas)
//         .and(left_top.rows())
//         .and(right_bottom.rows())
//         .for_each(|area, lt, rb| {
//             let width = (rb[0] - lt[0]).max(0.0);
//             let height = (rb[1] - lt[1]).max(0.0);
//             *area = width * height;
//         });
//     areas
// }

// fn iou_of(boxes0: &Array2<f32>, boxes1: &Array2<f32>, eps: f32) -> Array1<f32> {
//     let overlap_left_top = boxes0.slice(s![.., 0..2]).mapv(|x| x.max(boxes1[[0, 0]]));
//     let overlap_right_bottom = boxes0.slice(s![.., 2..4]).mapv(|x| x.min(boxes1[[0, 2]]));

//     let overlap_area = area_of(&overlap_left_top, &overlap_right_bottom);
//     let area0 = area_of(&boxes0.slice(s![.., 0..2]), &boxes0.slice(s![.., 2..4]));
//     let area1 = area_of(&boxes1.slice(s![.., 0..2]), &boxes1.slice(s![.., 2..4]));

//     overlap_area.clone() / (area0 + area1 - overlap_area.clone() + eps)
// }

// fn hard_nms(
//     box_scores: &Array2<f32>,
//     iou_threshold: f32,
//     top_k: isize,
//     candidate_size: usize,
// ) -> Array2<f32> {
//     // Ensure the function doesn't panic if box_scores is empty
//     if box_scores.is_empty() {
//         return Array2::<f32>::zeros((0, 5));
//     }

//     let mut picked = Vec::new();
//     let mut indexes: Vec<usize> = (0..box_scores.nrows()).collect();

//     // Sort indexes by scores descending
//     indexes.sort_by(|&a, &b| {
//         let score_a = box_scores[[a, 4]];
//         let score_b = box_scores[[b, 4]];
//         score_b
//             .partial_cmp(&score_a)
//             .unwrap_or(std::cmp::Ordering::Equal)
//     });

//     // Limit to candidate_size if necessary
//     indexes.truncate(candidate_size);

//     for &current in indexes.iter() {
//         if picked.len() as isize == top_k {
//             break;
//         }

//         let current_box = box_scores.slice(s![current, ..4]);

//         let mut is_max = true;

//         for &picked_index in &picked {
//             let picked_box = box_scores.slice(s![picked_index..picked_index + 1, ..4]);
//             let iou = iou_of(&picked_box, &current_box, iou_threshold);

//             if iou[0] > iou_threshold {
//                 is_max = false;
//                 break;
//             }
//         }

//         if is_max {
//             picked.push(current);
//         }
//     }

//     // Construct the result from picked indices
//     let mut results = Array2::<f32>::zeros((picked.len(), 5));
//     for (mut result_row, &pick) in results.outer_iter_mut().zip(picked.iter()) {
//         result_row.assign(&box_scores.slice(s![pick, ..]));
//     }
//     results
// }
#[show_image::main]
fn main() -> Result<(), anyhow::Error>{
    let _ = ort::init()
        .with_execution_providers([CPUExecutionProvider::default().build()])
        .commit();

    // let original_img = image::open(r"C:\Users\anand\.rust\webcamcapture\me.jpeg").unwrap();
    let original_img = image::open(r"C:\Users\anand\src\cpp\onnxtest\me.jpg").unwrap();
    let input = preprocess(original_img.clone().as_mut_rgb8().unwrap());
    let session = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .with_intra_threads(4)
        .unwrap()
        .commit_from_file(r"C:\Users\anand\.rust\webcamcapture\version-RFB-320.onnx")
        .unwrap();

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

    let mut durations = Vec::new();

    for _ in 0..10 {
        // Start timing before the operation
        let start = Instant::now();

        // Execute the operation you want to time
        let outputs = session
            .run(inputs!["input" => input.view()].unwrap())
            .unwrap();

        // Calculate the duration it took to perform the operation
        let duration = start.elapsed();
        durations.push(duration);
    }

    // Sort durations
    durations.sort();

    // Print sorted durations
    println!("Sorted durations:");
    for duration in durations {
        println!("{:?}", duration);
    }

    let outputs = session
        .run(inputs!["input" => input.view()].unwrap())
        .unwrap();

    // You can print out the duration or use it in other parts of your code

    let scores = outputs["scores"]
        .try_extract_tensor::<f32>()
        .unwrap()
        .t()
        .into_owned();
    let boxes = outputs["boxes"]
        .try_extract_tensor::<f32>()
        .unwrap()
        .t()
        .into_owned();
    let boxes = boxes.permuted_axes(vec![2, 1, 0]);
    println!("Scores: {:?}", scores);
    println!("Boxes: {:?}", boxes.shape());
    let selected = boxes.slice(s![0, 1, ..]);
    println!("Boxes: {:?}", selected);

    // [-0.0038034674, -0.002330102, 0.029901488, 0.06908778]
    // [-0.00378056, -0.00232621,  0.02996315,  0.06921531]

    // read "C:\Users\anand\src\cpp\onnxtest\ultraface\pimg.npy"
    let pimg: Array4<f32> =
        ndarray_npy::read_npy(r"C:\Users\anand\src\cpp\onnxtest\ultraface\pimg.npy").unwrap();
    println!("pimg: {:?}", pimg);
    println!("pimg: {:?}", pimg.shape());

    let outputs = session
        .run(inputs!["input" => pimg.view()].unwrap())
        .unwrap();

    let scores = outputs["scores"]
        .try_extract_tensor::<f32>()
        .unwrap()
        .t()
        .into_owned();
    let boxes = outputs["boxes"]
        .try_extract_tensor::<f32>()
        .unwrap()
        .t()
        .into_owned();

    let boxes = boxes.permuted_axes(vec![2, 1, 0]);
    let selected = boxes.slice(s![0, 1, ..]);
    println!("Boxes: {:?}", selected);
    // println!("Scores: {:?}", scores);
    //  [-0.0037805587, -0.002326209, 0.029963147, 0.06921531]
    //  [-0.00378056,   -0.00232621,  0.02996315,  0.06921531]
    // let img = original_img.;
    let me = image::open(r"C:\Users\anand\.rust\webcamcapture\me.jpeg")?;
    
    let window = create_window("image", Default::default())?;
    window.set_image("image-001", me)?;

    Ok(())
}
