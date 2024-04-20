use image::imageops::FilterType;
use image::io::Reader as ImageReader;
use image::{
    imageops, DynamicImage, GenericImageView, ImageBuffer, ImageFormat, ImageOutputFormat, RgbImage,
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

use ort::{self, inputs, CPUExecutionProvider, CUDAExecutionProvider, GraphOptimizationLevel};
use ort::{Session, TensorElementType, ValueType};
use raqote::{DrawOptions, DrawTarget};
use std::cmp::Ordering;
use std::env;
use std::error::Error;

const WIDTH: usize = 320;
const HEIGHT: usize = 240;
// const WIDTH: usize = 1000;
// const HEIGHT: usize = 480;
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
    // let new_img: RgbImage =
    //     ImageBuffer::from_raw(width, height, arr_data.clone().into_raw_vec()).expect("Invalid data length");
    // new_img
    //     .save("resized.png")
    //     .expect("Error saving resized image");

    let normalized = arr_data.mapv(|x| (x as f32 - 127.0) / 128.0);
    let transposed = normalized.permuted_axes([2, 0, 1]);
    let expanded = transposed.insert_axis(Axis(0));

    expanded
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut cam = initialize_camera()?;
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(env::current_dir()?.join("version-RFB-320.onnx"))?;

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
    let frame = cam.frame()?;

    let mut window = create_window("Raqote", WIDTH, HEIGHT)?;
    
    let size = window.get_size();
    let mut dt = DrawTarget::new(size.0 as i32, size.1 as i32);
    
    while window.is_open() && !window.is_key_down(Key::Escape) {
        if let Ok(frame) = cam.frame() {
            let img = frame.decode_image::<RgbFormat>().unwrap(); // Assumes successful decoding
                                                                  // Resize the image

            // let img = image::open(r"C:/Users/anand/src/cpp/onnxtest/me.jpg")?.to_rgb8();
            let (orig_width, orig_height) = img.dimensions();
            let rimg = imageops::resize(&img, WIDTH as u32, HEIGHT as u32, imageops::Lanczos3);

            let input = preprocess(&rimg);
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

            scores = scores.index_axis(Axis(0), 0).to_owned();
            boxes = boxes.index_axis(Axis(0), 0).to_owned();
            let cs = scores.slice(s![.., 1]);
            let mask = cs.mapv(|x| x > PROB_THRESHOLD);
            // np.where(mask)[0]
            // array([3903, 3943, 3945, 3983, 4270, 4290, 4291, 4292, 4293], dtype=int64)

            let indices: Vec<_> = mask
                .iter()
                .enumerate()
                .filter_map(|(i, &m)| if m { Some(i) } else { None })
                .collect();

            // print indices shape
            println!("Indices: {:?}", indices);

            // let filtered_probs = cs[indices.clone()];
            let filtered_probs = cs.select(Axis(0), &indices);
            let subset_boxes = // boxes[indices, ..];
        indices.iter().map(|&i| boxes.index_axis(Axis(0), i)).collect::<Vec<_>>();
            println!("Subset boxes: {:?}", subset_boxes);
            println!("scores: {:?}", scores.shape());
            println!("Boxes: {:?}", boxes.shape());

            println!("Mask: {:?}", filtered_probs.shape());
            // box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            println!("Filtered probs: {:?}", filtered_probs);

            let box_probs = Array::from_shape_fn((indices.len(), 5), |(i, j)| {
                if j == 4 {
                    filtered_probs[i]
                } else {
                    subset_boxes[i][j]
                }
            });

            // let mut picked_box_probs = vec![];
            // println!("Box probs: {:?}", box_probs);
            // println!("Box probs: {:?}", box_probs.shape());

            // box_probs = hard_nms(box_probs,
            //    iou_threshold=iou_threshold,
            //    top_k=top_k,
            //    )
            let bboxes_idxs = hard_nms(&box_probs, IOU_THRESHOLD, TOP_K, 200);
            // println!("BBoxes_idxs: {:?}", bboxes_idxs);
            let bboxes = box_probs.select(Axis(0), &bboxes_idxs);
            // println!("BBoxes: {:?}", bboxes);

            // picked_box_probs.append(box_probs)
            //     picked_box_probs[:, 0] *= width
            // picked_box_probs[:, 1] *= height
            // picked_box_probs[:, 2] *= width
            // picked_box_probs[:, 3] *= height
            // let real_boxes = Array::zeros((bboxes.shape()[0], 4));
            let ls = bboxes.slice(s![.., 1]).mapv(|x| x * WIDTH as f32);
            let ts = bboxes.slice(s![.., 0]).mapv(|x| x * HEIGHT as f32);
            let rs = bboxes.slice(s![.., 3]).mapv(|x| x * WIDTH as f32);
            let bs = bboxes.slice(s![.., 2]).mapv(|x| x * HEIGHT as f32);
            let real_boxes = Array::from_shape_fn((bboxes.shape()[0], 4), |(i, j)| match j {
                0 => ls[i],
                1 => ts[i],
                2 => rs[i],
                3 => bs[i],
                _ => unreachable!(),
            })
            .mapv(|x| x as usize);
            println!("Real boxes: {:?}", real_boxes);
            // draw boxes onto image
            // let mut rimg = rimg.clone();
            // for bbox in real_boxes.outer_iter() {
            //     let lt = (bbox[0], bbox[1]);
            //     let rb = (bbox[2], bbox[3]);
            //     image::imageops::draw_hollow_rect_mut(&mut rimg, lt, rb, image::Rgb([255, 0, 0]));
            // }
                // raqote::d
            // raqote::Image::

            let buffer: Vec<u32> = rimg
                .enumerate_pixels()
                .map(|(_, _, pixel)| {
                    let [r, g, b] = pixel.0;
                    (0 << 24)| ((r as u32) << 16) | ((g as u32) << 8) | (b as u32)
                })
                .collect();

            let draw_img = raqote::Image {
                width: WIDTH as i32,
                height: HEIGHT as i32,
                data: &buffer,
            };

            dt.draw_image_at(0., 0., &draw_img, &DrawOptions::default());

            // window.update_with_buffer(&buffer, WIDTH, HEIGHT)?;
            window.update_with_buffer(dt.get_data(), size.0, size.1).unwrap();
            // dt.draw_glyphs(font, point_size, ids, positions, src, options)

            // break;
        }
    }

    cam.stop_stream()?;
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
        println!("IOUs: {:?}", ious);
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
