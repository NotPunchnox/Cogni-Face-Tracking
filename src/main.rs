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

// ============================================
// FONCTIONS D'ANALYSE ET CALCUL
// ============================================

/// Analyse la position du visage par rapport au centre du frame
/// 
/// Retourne une structure FacePosition contenant:
/// - La position qualitative (gauche/centre/droite, haut/centre/bas)
/// - La distance en pixels du centre
/// - Un score de confiance (0.0 = mal centré, 1.0 = parfait centrage)
fn analyze_face_position(face_rect: &core::Rect, frame_width: i32, frame_height: i32) -> FacePosition {
    let center_x = frame_width / 2;
    let center_y = frame_height / 2;
    
    // Calculer le centre du visage détecté
    let face_center_x = face_rect.x + face_rect.width / 2;
    let face_center_y = face_rect.y + face_rect.height / 2;
    
    // Vecteur de distance depuis le centre
    let distance_x = face_center_x - center_x;
    let distance_y = face_center_y - center_y;
    
    // ALGORITHME: Détermination de la position horizontale
    // Utilise un seuil de 50px pour éviter les tremblements
    let horizontal = if distance_x < -50 {
        "GAUCHE".to_string()
    } else if distance_x > 50 {
        "DROITE".to_string()
    } else {
        "CENTRE".to_string()
    };
    
    // ALGORITHME: Détermination de la position verticale
    let vertical = if distance_y < -50 {
        "HAUT".to_string()
    } else if distance_y > 50 {
        "BAS".to_string()
    } else {
        "CENTRE".to_string()
    };
    
    // ALGORITHME: Calcul de la confiance de centrage
    // Distance maximale possible du centre à un coin du frame
    let max_distance = ((center_x * center_x + center_y * center_y) as f32).sqrt();
    // Distance actuelle du centre du visage à l'axe central
    let current_distance = ((distance_x * distance_x + distance_y * distance_y) as f32).sqrt();
    // Normaliser entre 0 et 1 (1 = centrage parfait)
    let confidence = (1.0 - (current_distance / max_distance)).max(0.0).min(1.0);
    
    FacePosition {
        horizontal,
        vertical,
        distance_x,
        distance_y,
        confidence,
    }
}

/// Génère les instructions de mouvement pour recentrer le visage
/// 
/// Utilise un facteur de lissage pour éviter les mouvements brusques
/// et implémenter une forme de filtre passe-bas
fn generate_movement_instruction(face_pos: &FacePosition, smoothing_factor: f32) -> MovementInstruction {
    // CONVERSION: pixels → degrés
    // Une caméra typique a un champ de vision de ~60°
    // Avec une résolution de ~640px, cela donne ~0.1°/pixel
    let degrees_per_pixel = 0.1;
    
    // CALCUL: Instructions de mouvement
    // Le smoothing_factor agit comme un filtre pour éviter les micromovements
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

/// Dessine les informations du visage et les instructions sur le frame
fn draw_face_info(
    frame: &mut Mat,
    face_rect: &core::Rect,
    face_pos: &FacePosition,
    instruction: &MovementInstruction,
) -> Result<(), Box<dyn std::error::Error>> {
    let thickness = 2;
    
    // Choisir la couleur du rectangle en fonction de la confiance de centrage
    let color = match face_pos.confidence {
        c if c > 0.8 => core::Scalar::new(0.0, 255.0, 0.0, 0.0),  // Vert = bien centré
        c if c > 0.5 => core::Scalar::new(0.0, 255.0, 255.0, 0.0), // Jaune = centrage moyen
        _ => core::Scalar::new(0.0, 0.0, 255.0, 0.0),              // Rouge = mal centré
    };
    
    // Dessiner le rectangle autour du visage
    imgproc::rectangle(frame, *face_rect, color, thickness, imgproc::LINE_8, 0)?;
    
    // Marquer le centre du visage avec un point bleu
    let face_center = core::Point::new(
        face_rect.x + face_rect.width / 2,
        face_rect.y + face_rect.height / 2,
    );
    imgproc::circle(frame, face_center, 5, core::Scalar::new(255.0, 0.0, 0.0, 0.0), -1, imgproc::LINE_8, 0)?;
    
    // Marquer le centre du frame avec un cercle vert
    let frame_width = frame.cols();
    let frame_height = frame.rows();
    imgproc::circle(frame, core::Point::new(frame_width / 2, frame_height / 2), 8, core::Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, 0)?;
    
    // Afficher les instructions de mouvement
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
    
    // Afficher le pourcentage de confiance
    let confidence_text = format!("Confiance: {:.1}%", face_pos.confidence * 100.0);
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

// ============================================
// FONCTION PRINCIPALE
// ============================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════════════════╗");
    println!("║  SYSTÈME DE TRACKING DE VISAGE AVEC CENTRAGE AUTO      ║");
    println!("╚════════════════════════════════════════════════════════╝\n");

    let mut controller = ST3215::new("/dev/ttyACM0")?;

    // Charger le classifieur Haar Cascade pour la détection de visages
    let mut face_cascade = objdetect::CascadeClassifier::new(
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
    )?;

    // Créer la fenêtre pour visualiser la vidéo
    let window = "Face Tracking - Centrage Automatique";
    highgui::named_window(window, highgui::WINDOW_AUTOSIZE)?;

    // Récupérer le flux vidéo depuis la caméra réseau
    let mut cam = videoio::VideoCapture::from_file("http://192.168.1.48:8000/stream.mjpg", videoio::CAP_ANY)?;
    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!(" Impossible d'ouvrir la caméra! Vérifiez: http://192.168.1.48:8000/stream.mjpg");
    }
    
    println!("✓ Caméra connectée avec succès");
    
    // PARAMÈTRES DE TRACKING
    // Le smoothing_factor (0 à 1) contrôle la réactivité du système:
    // - 0.1 = très lissé (mouvements lents)
    // - 0.5 = équilibre bruit/réactivité
    // - 1.0 = pas de lissage (peut être saccadé)
    let smoothing_factor = 0.5;
    
    // Positions précédentes du servomoteur (pour hystérésis)
    let mut last_pan_pos = 90u16;   // 90° = position neutre/centrale
    let mut last_tilt_pos = 90u16;
    
    // Boucle principale
    loop {
        let mut frame = Mat::default();
        cam.read(&mut frame)?;
        if frame.empty() { break; }

        let frame_width = frame.cols();
        let frame_height = frame.rows();

        // Convertir en nuances de gris pour l'algorithme de détection
        let mut gray = Mat::default();
        imgproc::cvt_color(&frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        // Détecter les visages avec le classifieur Cascade
        let mut faces = core::Vector::new();
        face_cascade.detect_multi_scale(
            &gray, 
            &mut faces, 
            1.1,                                      // Facteur d'échelle
            4,                                        // Voisins minimum
            0,                                        // Drapeaux
            core::Size::new(30, 30),                  // Taille min
            core::Size::new(0, 0)                     // Taille max
        )?;

        if faces.len() > 0 {
            // Prendre le premier/plus grand visage détecté
            let face = faces.get(0)?;
            
            // ÉTAPE 1: Analyser la position du visage
            let face_pos = analyze_face_position(&face, frame_width, frame_height);
            
            // ÉTAPE 2: Générer les instructions de mouvement
            let instruction = generate_movement_instruction(&face_pos, smoothing_factor);
            
            // ÉTAPE 3: Afficher les informations sur le frame
            draw_face_info(&mut frame, &face, &face_pos, &instruction)?;
            
            // ÉTAPE 4: Afficher dans la console
            println!("\n╔════════════════════════════════════╗");
            println!("║  VISAGE DÉTECTÉ ET ANALYSÉ    ║");
            println!("╠════════════════════════════════════╣");
            println!("║ Position: {} / {}", face_pos.horizontal, face_pos.vertical);
            println!("║ Distance du centre: X={:+}px, Y={:+}px", face_pos.distance_x, face_pos.distance_y);
            println!("║ Confiance de centrage: {:.1}%", face_pos.confidence * 100.0);
            println!("║ Instructions: {}", instruction.description);
            
            // ÉTAPE 5: Générer les commandes de mouvement
            if instruction.pan.abs() > 0.1 || instruction.tilt.abs() > 0.1 {
                println!("╠════════════════════════════════════╣");
                println!("║ COMMANDES À ENVOYER:           ║");
                
                // Commandes horizontales (Pan)
                if instruction.pan < -1.0 {
                    println!("║   ← Tourner de {:.1}° à GAUCHE", -instruction.pan);
                } else if instruction.pan > 1.0 {
                    println!("║   → Tourner de {:.1}° à DROITE", instruction.pan);
                }
                
                // Commandes verticales (Tilt)
                if instruction.tilt < -1.0 {
                    println!("║   ↑ Tourner de {:.1}° vers le HAUT", -instruction.tilt);
                } else if instruction.tilt > 1.0 {
                    println!("║   ↓ Tourner de {:.1}° vers le BAS", instruction.tilt);
                }
                
                println!("╚════════════════════════════════════╝");
                
                // Calcul des positions servomoteur
                // Convention: 90° = position neutre, min=0°, max=180°
                let mut new_pan_pos = ((90.0 + instruction.pan).max(0.0).min(180.0)) as u16;
                let new_tilt_pos = ((90.0 + instruction.tilt).max(0.0).min(180.0)) as u16;

                new_pan_pos = (new_pan_pos * 4096) / 360;
                
                println!("\nPositions servomoteur calculées:");
                println!("   Pan:  {}° → {}° (delta: {:.1}°)", last_pan_pos, new_pan_pos, instruction.pan);
                println!("   Tilt: {}° → {}° (delta: {:.1}°)", last_tilt_pos, new_tilt_pos, instruction.tilt);
                
                // INTÉGRATION SERVOMOTEUR
                
                   controller.move_to(1, new_pan_pos, 2400, 50, false);
                //    controller.set_servo_position(2, new_tilt_pos)?;  // Servo 2 = Tilt (vertical)

                // 4. Optionnel - Attendre que les servos réagissent:
                   std::thread::sleep(std::time::Duration::from_millis(50));
                
                last_pan_pos = new_pan_pos;
                last_tilt_pos = new_tilt_pos;
            } else {
                println!("║ ✓ Visage bien centré!");
                println!("╚════════════════════════════════════╝");
            }
        } else {
            println!("  Aucun visage détecté - recherche en cours...");
            
            let text = "Aucun visage detecte";
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

        // Afficher l'image dans la fenêtre
        highgui::imshow(window, &frame)?;
        
        // Attendre 1ms pour permettre à OpenCV de traiter les événements
        // Appuyer sur n'importe quelle touche pour quitter
        let key = highgui::wait_key(1)?;
        if key > 0 {
            println!("\n✓ Arrêt demandé par l'utilisateur");
            break;
        }
    }

    println!("\n✓ Programme terminé correctement");
    Ok(())
}
