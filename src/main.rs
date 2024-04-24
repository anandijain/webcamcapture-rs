use image::imageops::FilterType;
use image::io::Reader as ImageReader;
use image::{
    imageops, DynamicImage, GenericImageView, ImageBuffer, ImageFormat, ImageOutputFormat, Rgb,
    RgbImage,
};
use minifb::{Key, Window, WindowOptions};
use ndarray::prelude::*;
use ndarray::{s, Array, Array1, Array2, Array4, ArrayBase, Axis, Data, Ix2, OwnedRepr, Zip};
use ndarray_npy::write_npy;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::{
    utils::{frame_formats, CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};

use ort::{
    self, inputs, CPUExecutionProvider, CUDAExecutionProvider, DirectMLExecutionProvider,
    GraphOptimizationLevel, SessionOutputs,
};
use ort::{Session, TensorElementType, ValueType};
use raqote::{DrawOptions, DrawTarget, PathBuilder, SolidSource, Source};
use std::cmp::Ordering;
use std::env;
use std::error::Error;
use std::time::{Duration, Instant};

// const WIDTH: usize = 320;
// const HEIGHT: usize = 240;
const WIDTH: usize = 640;
const HEIGHT: usize = 480;
const IOU_THRESHOLD: f32 = 0.5;
const TOP_K: isize = -1;
const PROB_THRESHOLD: f32 = 0.7;
const IOU_EPS: f32 = 1e-5;

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
    let resized = imageops::resize(img, 320, 240, FilterType::Nearest);
    let (width, height) = resized.dimensions();
    let raw_data = resized.into_raw(); // This gives us Vec<u8> of the data
    let arr_data = Array::from_shape_vec((height as usize, width as usize, 3), raw_data)
        .expect("Error converting raw buffer to ndarray");
    let normalized = arr_data.mapv(|x| (x as f32 - 127.0) / 128.0);
    let transposed = normalized.permuted_axes([2, 0, 1]);
    let expanded = transposed.insert_axis(Axis(0));

    expanded
}

fn display_model_inputs_outputs(session: &Session) {
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
}
/*
Inputs:
    0 input: Tensor<f32>(1, 3, 240, 320)
Outputs:
    0 scores: Tensor<f32>(1, 4420, 2)
    1 boxes: Tensor<f32>(1, 4420, 4)
*/
fn prep_img_and_infer(
    img: &RgbImage,
    session: &Session,
) -> Result<(ArrayD<f32>, ArrayD<f32>), anyhow::Error> {
    let input = preprocess(&img);

    let outputs = session.run(inputs!["input" => input.view()]?)?;
    let mut scores = outputs["scores"]
        .try_extract_tensor::<f32>()?
        .t()
        .into_owned();
    let mut boxes = outputs["boxes"]
        .try_extract_tensor::<f32>()?
        .t()
        .into_owned();
    boxes = boxes.permuted_axes(vec![2, 1, 0]);
    scores = scores.permuted_axes(vec![2, 1, 0]);
    boxes = boxes.index_axis(Axis(0), 0).to_owned();
    scores = scores.index_axis(Axis(0), 0).to_owned();
    Ok((boxes, scores))
}

fn box_probs_from_outputs(boxes: &ArrayD<f32>, scores: &ArrayD<f32>) -> Array2<f32> {
    let cs = scores.slice(s![.., 1]);
    let mask = cs.mapv(|x| x > PROB_THRESHOLD);

    let indices: Vec<_> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i) } else { None })
        .collect();

    let filtered_probs = cs.select(Axis(0), &indices);
    let subset_boxes = boxes.select(Axis(0), &indices);

    let box_probs = Array::from_shape_fn((indices.len(), 5), |(i, j)| {
        if j == 4 {
            filtered_probs[i]
        } else {
            subset_boxes[[i, j]]
        }
    });
    box_probs
}

fn float_bboxes_to_int(bboxes: Array2<f32>, width: usize, height: usize) -> Array2<u32> {
    let ls = bboxes.slice(s![.., 0]).mapv(|x| (x * width as f32) as u32);
    let ts = bboxes.slice(s![.., 1]).mapv(|x| (x * height as f32) as u32);
    let rs = bboxes.slice(s![.., 2]).mapv(|x| (x * width as f32) as u32);
    let bs = bboxes.slice(s![.., 3]).mapv(|x| (x * height as f32) as u32);
    let real_boxes = Array::from_shape_fn((bboxes.shape()[0], 4), |(i, j)| match j {
        0 => ls[i],
        1 => ts[i],
        2 => rs[i],
        3 => bs[i],
        _ => unreachable!(),
    });
    real_boxes
}

fn preproc_styletransfer(img: &RgbImage, session: &Session) -> Result<Array4<f32>, anyhow::Error> {
    // let (orig_width, orig_height) = img.dimensions();

    // assume that there is a single input and that it is in (B, C, H, W) format or BCWH idk but that channel is [1]
    let sess_in = session.inputs.iter().next().expect("No inputs found");
    let name = sess_in.name.clone();
    let dims = sess_in.input_type.tensor_dimensions().unwrap();
    let (width, height) = (dims[3] as u32, dims[2] as u32);
    let rimg = imageops::resize(img, width, height, image::imageops::FilterType::Nearest);
    let raw_data = rimg.into_raw(); // This gives us Vec<u8> of the data
    let arr_data = Array::from_shape_vec((height as usize, width as usize, 3), raw_data.clone())
        .expect("Error converting raw buffer to ndarray");
    let f_arr = arr_data.mapv(|x| x as f32);
    let transposed = f_arr.permuted_axes([2, 0, 1]);
    let expanded = transposed.insert_axis(Axis(0));
    Ok(expanded)
    // Ok(inputs![name => expanded.view()]?) // ideally preproc always returns sessioninputs
}

fn postproc_styletransfer(
    out: &SessionOutputs,
    orig_width: u32,
    orig_height: u32,
) -> Result<RgbImage, anyhow::Error> {
    // assuming output is in BCHW format

    let out = out["output1"].try_extract_tensor::<f32>()?.into_owned();
    let out2 = out.into_dimensionality::<ndarray::Ix4>().unwrap();
    let (_b, _c, h, w) = out2.dim();

    let o = out2
        .mapv(|x| x.max(0.0).min(255.0) as u8)
        .index_axis(Axis(0), 0)
        .to_owned();

    let mut out_img: RgbImage = ImageBuffer::new(h as u32, w as u32);
    // Iterate over each pixel position
    for y in 0..h {
        for x in 0..w {
            let (xx, yy) = (x as usize, y as usize);
            let (r, g, b) = (o[[0, yy, xx]], o[[1, yy, xx]], o[[2, yy, xx]]);
            out_img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }
    let out_reshaped = imageops::resize(
        &out_img,
        orig_width,
        orig_height,
        imageops::FilterType::CatmullRom,
    );
    Ok(out_reshaped)
}

fn imgbuf_to_buf(img: &RgbImage) -> Vec<u32> {
    let buffer: Vec<u32> = img
        .enumerate_pixels()
        .map(|(_, _, pixel)| {
            let [r, g, b] = pixel.0;
            (0 << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
        })
        .collect();
    buffer
}
fn new_capture_predict_loop(cam: &mut Camera, session: &Session) -> Result<(), anyhow::Error> {
    let r = cam.resolution();
    let (cam_width, cam_height) = (r.width(), r.height());
    let mut window = create_window("Raqote", cam_width as usize, cam_height as usize)?;
    let size = window.get_size();
    let (orig_width, orig_height) = size;
    let mut dt = DrawTarget::new(size.0 as i32, size.1 as i32);

    let mut frame_count = 0;
    let mut last_fps_time = Instant::now();
    let fps_interval = Duration::new(1, 0); // 1 second

    while window.is_open() && !window.is_key_down(Key::Escape) {
        if let Ok(frame) = cam.frame() {
            let mut img = frame.decode_image::<RgbFormat>().unwrap(); // Assumes successful decoding
            let input = preproc_styletransfer(&img, session)?;
            let pred = session.run(inputs!["input1" => input.view()]?)?;
            let out_img = postproc_styletransfer(&pred, orig_width as u32, orig_height as u32)?;
            let out_buf = imgbuf_to_buf(&out_img);
            window.update_with_buffer(&out_buf, size.0, size.1).unwrap();
        }
    }
    Ok(())
}



fn capture_predict_loop(cam: &mut Camera, session: &Session) -> Result<(), anyhow::Error> {
    let r = cam.resolution();
    let (cam_width, cam_height) = (r.width(), r.height());
    let mut window = create_window("Raqote", cam_width as usize, cam_height as usize)?;
    let size = window.get_size();
    let (orig_width, orig_height) = size;
    let mut dt = DrawTarget::new(size.0 as i32, size.1 as i32);

    let mut frame_count = 0;
    let mut last_fps_time = Instant::now();
    let fps_interval = Duration::new(1, 0); // 1 second

    while window.is_open() && !window.is_key_down(Key::Escape) {
        if let Ok(frame) = cam.frame() {
            frame_count += 1;

            let mut img = frame.decode_image::<RgbFormat>().unwrap(); // Assumes successful decoding
            let rimg = imageops::resize(&img, WIDTH as u32, HEIGHT as u32, imageops::Lanczos3);
            let (boxes, scores) = prep_img_and_infer(&rimg, session)?;
            let box_probs = box_probs_from_outputs(&boxes, &scores);
            let bboxes_idxs = hard_nms(&box_probs, IOU_THRESHOLD, TOP_K, 200);
            let bboxes = box_probs.select(Axis(0), &bboxes_idxs);

            let pix_bboxes =
                float_bboxes_to_int(bboxes.clone(), cam_width as usize, cam_height as usize);

            // println!("BBoxes: {:?}", pix_bboxes);
            for i in 0..pix_bboxes.shape()[0] {
                draw_rect(
                    &mut img,
                    &pix_bboxes.index_axis(Axis(0), i).to_owned(),
                    Rgb([255, 255, 255]),
                    5,
                );
            }
            let buffer: Vec<u32> = img
                .enumerate_pixels()
                .map(|(_, _, pixel)| {
                    let [r, g, b] = pixel.0;
                    (0 << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
                })
                .collect();

            window.update_with_buffer(&buffer, size.0, size.1).unwrap();
            if last_fps_time.elapsed() >= fps_interval {
                let fps = frame_count as f32 / last_fps_time.elapsed().as_secs_f32();
                println!("FPS: {}", fps);
                frame_count = 0;
                last_fps_time = Instant::now();
            }
        }
    }
    Ok(())
}

// Function to draw a rectangle on the image
fn draw_rect(img: &mut RgbImage, bbox: &Array1<u32>, color: Rgb<u8>, line_width: u32) {
    let (img_width, img_height) = img.dimensions();

    // Extracting bounding box coordinates
    let left = bbox[0] as i32;
    let top = bbox[1] as i32;
    let right = bbox[2] as i32;
    let bottom = bbox[3] as i32;

    // Drawing the rectangle
    for w in 0..line_width as i32 {
        // Horizontal lines
        for x in left..=right {
            if x >= 0 && x < img_width as i32 {
                if top + w >= 0 && top + w < img_height as i32 {
                    img.put_pixel(x as u32, (top + w) as u32, color);
                }
                if bottom - w >= 0 && bottom - w < img_height as i32 {
                    img.put_pixel(x as u32, (bottom - w) as u32, color);
                }
            }
        }
        // Vertical lines
        for y in top..=bottom {
            if y >= 0 && y < img_height as i32 {
                if left + w >= 0 && left + w < img_width as i32 {
                    img.put_pixel((left + w) as u32, y as u32, color);
                }
                if right - w >= 0 && right - w < img_width as i32 {
                    img.put_pixel((right - w) as u32, y as u32, color);
                }
            }
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let providers = [
        // DirectMLExecutionProvider::default().build(),
        CUDAExecutionProvider::default().build(),
    ];
    ort::init().with_execution_providers(&providers).commit()?;

    let session = Session::builder()?
        .with_execution_providers(providers)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        // .with_intra_threads(4)?
        // .commit_from_file(env::current_dir()?.join("version-RFB-320.onnx"))?;
        .commit_from_file(env::current_dir()?.join("mosaic-9.onnx"))?;

    println!("{:?}", session);

    display_model_inputs_outputs(&session);
    // panic!();
    let mut img = image::open(r"C:/Users/anand/src/cpp/onnxtest/me.jpg")?.to_rgb8();
    println!("Image dimensions: {:?}", img.dimensions());

    let mut cam = initialize_camera()?;
    // let frame = cam.frame()?;
    new_capture_predict_loop(&mut cam, &session);

    let mut rimg = imageops::resize(&img, WIDTH as u32, HEIGHT as u32, imageops::Lanczos3);

    let (orig_width, orig_height) = img.dimensions();
    let (mut boxes, mut scores) = prep_img_and_infer(&img, &session)?;
    let box_probs = box_probs_from_outputs(&boxes, &scores);
    let bboxes_idxs = hard_nms(&box_probs, IOU_THRESHOLD, TOP_K, 200);
    let bboxes = box_probs.select(Axis(0), &bboxes_idxs);

    let pix_bboxes = float_bboxes_to_int(bboxes.clone(), orig_width as usize, orig_height as usize);
    println!("BBoxes: {:?}", pix_bboxes);
    for i in 0..pix_bboxes.shape()[0] {
        draw_rect(
            &mut img,
            &pix_bboxes.index_axis(Axis(0), i).to_owned(),
            Rgb([255, 255, 255]),
            5,
        );
    }

    img.save("drawn.png")?;
    Ok(())
}

// fn frame_to_imgbuf(b: Buffer) -> RgbImage {}
// fn predict()
fn hard_nms(
    box_probs: &Array2<f32>,
    iou_threshold: f32,
    top_k: isize,
    candidate_size: usize,
) -> Vec<usize> {
    let scores = box_probs.slice(s![.., 4]);
    let boxes = box_probs.slice(s![.., ..4]);
    // println!("Scores: {:?}", scores);
    // println!("Boxes: {:?}", boxes);
    //     picked = []
    // indexes = np.argsort(scores)
    let mut picked = vec![];
    let mut indexes = argsort(&scores.clone().to_owned());
    // println!("Indexes: {:?}", indexes);

    let foo = box_probs.select(Axis(0), &indexes.to_vec());
    // println!("Foo: {:?}", foo);
    // Foo:
    // [[0.49421334, 0.3905043, 0.6528523, 0.69107586, 0.71388346],
    //  [0.49335456, 0.40095496, 0.65837955, 0.7006705, 0.90883833],
    //  [0.4972741, 0.40358883, 0.65144116, 0.6924824, 0.92268896],
    //  [0.49552137, 0.39640325, 0.64830726, 0.69495136, 0.9628681],
    //  [0.49718904, 0.39342397, 0.65392995, 0.6966315, 0.9788382],
    //  [0.50078106, 0.3889392, 0.66095245, 0.7001719, 0.9992065],
    //  [0.49797508, 0.396097, 0.6569873, 0.7020888, 0.999954],
    //  [0.4988954, 0.394103, 0.65823495, 0.699032, 0.9999993],
    //  [0.49895865, 0.39731386, 0.65290815, 0.69771373, 0.9999993]],

    // indexes = indexes[-candidate_size:]
    // let start_idx = 0.max(indexes.len() - candidate_size);
    // let lindexts = indexes.slice(s![start_idx..]);
    // println!("LIndexes: {:?}", lindexts);
    // while len(indexes) > 0:
    while indexes.len() > 0 {
        //     current = indexes[-1]
        let current = indexes[indexes.len() - 1];

        //     picked.append(current)
        picked.push(current);
        //     if 0 < top_k == len(picked) or len(indexes) == 1:
        //         break
        if false < (top_k == picked.len() as isize) || indexes.len() == 1 {
            break;
        }
        //     current_box = boxes[current, :]
        let current_box = boxes.index_axis(Axis(0), current).to_owned();
        // println!("Current box: {:?}", current_box);
        //     indexes = indexes[:-1]
        indexes = indexes.slice(s![..indexes.len() - 1]).to_owned();
        // print num indexes
        // println!("Num indexes: {:?}", indexes.len());
        //     rest_boxes = boxes[indexes, :]
        let rest_boxes = indexes
            .iter()
            .map(|&i| boxes.index_axis(Axis(0), i))
            .collect::<Vec<_>>();
        // println!("Rest boxes: {:?}", rest_boxes);
        // println!("Rest boxes: {:?}", rest_boxes.len());

        //     iou = iou_of(
        //         rest_boxes,
        //         np.expand_dims(current_box, axis=0),
        //     )
        // let foo = current_box.insert_axis(Axis(0));
        // let b = boxes.index_axis(Axis(0), 0).to_owned();
        // let iou = iou_of(&b, &current_box);

        let ious = rest_boxes
            .iter()
            .map(|rb| iou_of(&rb.to_owned(), &current_box))
            .collect::<Vec<_>>();

        // println!("IOU : {:?}", iou);
        // println!("IOUs: {:?}", ious);
        //     indexes = indexes[iou <= iou_threshold]
        let iou_mask = Array::from_vec(ious).mapv(|x| x <= iou_threshold);
        let mask_idxs = iou_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(i) } else { None })
            .collect::<Vec<_>>();
        indexes = indexes.select(Axis(0), &mask_idxs);
    }
    // let foo = (3, 2, 4).f();
    // todo!()
    // Ok(())
    // box_scores[picked, :]
    // box_scores
    picked
}

fn argsort(arr: &Array1<f32>) -> Array1<usize> {
    let mut indices: Vec<usize> = (0..arr.len()).collect();
    indices.sort_by(|&i, &j| arr[i].partial_cmp(&arr[j]).unwrap_or(Ordering::Equal));
    Array::from_vec(indices)
}
// def iou_of(boxes0, boxes1, eps=1e-5):
//     """
//     Return intersection-over-union (Jaccard index) of boxes.
//     Args:
//         boxes0 (N, 4): ground truth boxes.
//         boxes1 (N or 1, 4): predicted boxes.
//         eps: a small number to avoid 0 as denominator.
//     Returns:
//         iou (N): IoU values.
//     """
//     overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
//     overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

//     overlap_area = area_of(overlap_left_top, overlap_right_bottom)
//     area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
//     area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
//     return overlap_area / (area0 + area1 - overlap_area + eps)

// returns the intersection left top and right bottom as a vector of length 4
fn box_intersection(a: &Array1<f32>, b: &Array1<f32>) -> Array1<f32> {
    // a.mapv(|x| x.max(0.0)) + b.mapv(|x| x.max(0.0))
    let a_lt = a.slice(s![..2]);
    let a_rb = a.slice(s![2..]);
    let b_lt = b.slice(s![..2]);
    let b_rb = b.slice(s![2..]);
    let overlap_lt = Zip::from(&a_lt).and(&b_lt).map_collect(|&a, &b| a.max(b));
    let overlap_rb = Zip::from(&a_rb).and(&b_rb).map_collect(|&a, &b| a.min(b));
    Array::from_shape_vec(
        4,
        vec![overlap_lt[0], overlap_lt[1], overlap_rb[0], overlap_rb[1]],
    )
    .expect("Error creating overlap array")
}
// iou of a single pair of bboxes
fn iou_of(boxes0: &Array1<f32>, boxes1: &Array1<f32>) -> f32 {
    // println!("Boxes0: {:?}", boxes0);
    // println!("Boxes1: {:?}", boxes1);
    let box_inter = box_intersection(boxes0, boxes1);
    // println!("Box inter: {:?}", box_inter);
    let overlap_area = area_of_vec(&box_inter);
    let area0 = area_of_vec(boxes0);
    let area1 = area_of_vec(boxes1);
    overlap_area / (area0 + area1 - overlap_area + IOU_EPS)
}

// def area_of(left_top, right_bottom):
//     """
//     Compute the areas of rectangles given two corners.
//     Args:
//         left_top (N, 2): left top corner.
//         right_bottom (N, 2): right bottom corner.
//     Returns:
//         area (N): return the area.
//     """
//     hw = np.clip(right_bottom - left_top, 0.0, None)
//     area = hw[..., 0] * hw[..., 1]
//     return area

fn area_of(left_top: Array2<f32>, right_bottom: Array2<f32>) -> Array1<f32> {
    let hw = Zip::from(&right_bottom)
        .and(&left_top)
        .map_collect(|&rb, &lt| rb - lt)
        .mapv(|x| x.max(0.0));
    // println!("HW: {:?}", hw);
    hw.map_axis(Axis(1), |x| x[0] * x[1])
}

fn area_of1(left_top: Array1<f32>, right_bottom: Array1<f32>) -> f32 {
    let hw = right_bottom - left_top;
    let hw = hw.mapv(|x| x.max(0.0));
    hw[0] * hw[1]
}
fn area_of_vec(bbox: &Array1<f32>) -> f32 {
    let hw = &bbox.slice(s![2..]) - &bbox.slice(s![..2]);
    let hw = hw.mapv(|x| x.max(0.0));
    hw[0] * hw[1]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_area_of() {
        // have the following equal array![9.0, 25.0]
        let left_top = array![[1., 1.], [0., 0.]];
        let right_bot = array![[4., 4.], [5., 5.]];
        assert_eq!(area_of(left_top, right_bot), array![9.0, 25.0]);
    }
}
