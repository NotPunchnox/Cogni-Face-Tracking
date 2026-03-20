use opencv::{core, imgproc, prelude::*};
use opencv::{highgui, videoio, objdetect};
use st3215::ST3215;


/// Représente la position du visage analysée
#[derive(Debug, Clone)]
struct FacePosition {
    horizontal: String, // "GAUCHE", "CENTRE", "DROITE"
    vertical: String,   // "HAUT", "CENTRE", "BAS"
    distance_x: i32,    // Pixels du centre (négatif = gauche, positif = droite)
    distance_y: i32,    // Pixels du centre (négatif = haut, positif = bas)
    confidence: f32,    // Confiance (0.0 à 1.0)
}

/// Instructions de mouvement du servomoteur
#[derive(Debug)]
struct MovementInstruction {
    pan: f32,   // Rotation horizontale en degrés (négatif = gauche, positif = droite)
    tilt: f32,  // Rotation verticale en degrés (négatif = haut, positif = bas)
    description: String,
}

/// Analyzes face position relative to frame center.
/// Returns horizontal/vertical position, pixel distance, and confidence score (0.0-1.0).
fn analyze_face_position(face_rect: &core::Rect, frame_width: i32, frame_height: i32) -> FacePosition {
    let center_x = frame_width / 2;
    let center_y = frame_height / 2;
    
    // Calculer le centre du visage détecté
    let face_center_x = face_rect.x + face_rect.width / 2;
    let face_center_y = face_rect.y + face_rect.height / 2;
    
    // Vecteur de distance depuis le centre
    let distance_x = face_center_x - center_x;
    let distance_y = face_center_y - center_y;
    
    let horizontal = if distance_x < -50 {
        "GAUCHE".to_string()
    } else if distance_x > 50 {
        "DROITE".to_string()
    } else {
        "CENTRE".to_string()
    };
    
    let vertical = if distance_y < -50 {
        "HAUT".to_string()
    } else if distance_y > 50 {
        "BAS".to_string()
    } else {
        "CENTRE".to_string()
    };
    
    // Confidence score: 1.0 = centered, 0.0 = at frame edge
    let max_distance = ((center_x * center_x + center_y * center_y) as f32).sqrt();
    let current_distance = ((distance_x * distance_x + distance_y * distance_y) as f32).sqrt();
    let confidence = (1.0 - (current_distance / max_distance)).max(0.0).min(1.0);
    
    FacePosition {
        horizontal,
        vertical,
        distance_x,
        distance_y,
        confidence,
    }
}

/// Generates servo movement instructions from face position.
/// Uses smoothing factor to prevent jerky motion (values 0.1-1.0).
fn generate_movement_instruction(face_pos: &FacePosition, smoothing_factor: f32) -> MovementInstruction {
    let degrees_per_pixel = 0.1;
    
    let pan = face_pos.distance_x as f32 * degrees_per_pixel * smoothing_factor;
    let tilt = face_pos.distance_y as f32 * degrees_per_pixel * smoothing_factor;
    
    let description = format!(
        "Position: {} | {} | Rotation: {:.1}° (pan) × {:.1}° (tilt)",
        face_pos.horizontal, face_pos.vertical, pan, tilt
    );
    
    MovementInstruction {
        pan,
        tilt,
        description,
    }
}

// ============================================
// FONCTIONS DE VISUALISATION
// ============================================

/// Draws face rectangle, center point, and confidence info on frame.
fn draw_face_info(
    frame: &mut Mat,
    face_rect: &core::Rect,
    face_pos: &FacePosition,
    instruction: &MovementInstruction,
) -> Result<(), Box<dyn std::error::Error>> {
    let thickness = 2;
    
    // Color by confidence: Green >80%, Yellow 50-80%, Red <50%
    let color = match face_pos.confidence {
        c if c > 0.8 => core::Scalar::new(0.0, 255.0, 0.0, 0.0),
        c if c > 0.5 => core::Scalar::new(0.0, 255.0, 255.0, 0.0),
        _ => core::Scalar::new(0.0, 0.0, 255.0, 0.0),
    };
    
    imgproc::rectangle(frame, *face_rect, color, thickness, imgproc::LINE_8, 0)?;
    
    let face_center = core::Point::new(
        face_rect.x + face_rect.width / 2,
        face_rect.y + face_rect.height / 2,
    );
    imgproc::circle(frame, face_center, 5, core::Scalar::new(255.0, 0.0, 0.0, 0.0), -1, imgproc::LINE_8, 0)?;
    
    let frame_width = frame.cols();
    let frame_height = frame.rows();
    imgproc::circle(frame, core::Point::new(frame_width / 2, frame_height / 2), 8, core::Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, 0)?;
    
    imgproc::put_text(
        frame,
        &instruction.description,
        core::Point::new(10, 30),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.7,
        core::Scalar::new(255.0, 255.0, 255.0, 0.0),
        2,
        imgproc::LINE_8,
        false,
    )?;
    
    let confidence_text = format!("Confidence: {:.1}%", face_pos.confidence * 100.0);
    imgproc::put_text(
        frame,
        &confidence_text,
        core::Point::new(10, 60),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.7,
        core::Scalar::new(255.0, 255.0, 255.0, 0.0),
        2,
        imgproc::LINE_8,
        false,
    )?;
    
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Face Tracking System\n");

    let mut controller = ST3215::new("/dev/ttyACM0")?;
    let mut face_cascade = objdetect::CascadeClassifier::new(
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    )?;

    let window = "Face Tracking";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;

    let mut cam = videoio::VideoCapture::from_file("http://192.168.1.48:8000/stream.mjpg", videoio::CAP_ANY)?;
    if !videoio::VideoCapture::is_opened(&cam)? {
        panic!("Camera connection failed");
    }
    
    let smoothing_factor = 0.5;
    let mut last_pan_pos = 2048u16;
    let mut last_tilt_pos = 2048u16;
    
    loop {
        let mut frame = Mat::default();
        cam.read(&mut frame)?;
        if frame.empty() { break; }

        let frame_width = frame.cols();
        let frame_height = frame.rows();

        let mut gray = Mat::default();
        imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        let mut faces = core::Vector::new();
        face_cascade.detect_multi_scale(
            &gray, 
            &mut faces, 
            1.1,
            4,
            0,
            core::Size::new(30, 30),
            core::Size::new(0, 0)
        )?;

        if faces.len() > 0 {
            let face: core::Rect_<i32> = faces.get(0)?;
            let face_pos: FacePosition = analyze_face_position(&face, frame_width, frame_height);
            let instruction: MovementInstruction = generate_movement_instruction(&face_pos, smoothing_factor);
            
            draw_face_info(&mut frame, &face, &face_pos, &instruction)?;
            
            if instruction.pan.abs() > 0.1 || instruction.tilt.abs() > 0.1 {
                let new_pan_pos = ((90.0 + instruction.pan).max(0.0).min(180.0) * 4096.0 / 360.0) as u16;
                let new_tilt_pos = ((90.0 - instruction.tilt).max(0.0).min(180.0) * 4096.0 / 360.0) as u16;
                
                let _ = controller.move_to(4, new_pan_pos, 2400, 50, false);
                
                last_pan_pos = new_pan_pos;
                last_tilt_pos = new_tilt_pos;
            }
        } else {
            let text = "No face detected";
            imgproc::put_text(
                &mut frame,
                text,
                core::Point::new(10, 30),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.7,
                core::Scalar::new(0.0, 0.0, 255.0, 0.0),
                2,
                imgproc::LINE_8,
                false,
            )?;
        }

        highgui::imshow(window, &frame)?;
        
        let key = highgui::wait_key(1).unwrap_or(-1);
        if key > 0 {
            break;
        }
        
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    Ok(())
}
