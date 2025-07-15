#!/usr/bin/env python3
"""genbreaks demo script"""

import sys
import os
import glob
import time

# add src to path
sys.path.append('src')


def run_audio_demo():
    print("++ genbreaks audio generation demo ++\n")
    
    # check for samples in breaks folder
    samples_dir = "breaks"
    if not os.path.exists(samples_dir):
        print(f"error: {samples_dir} folder not found!")
        print("please create a breaks folder with breakbeat wav files")
        return False
    
    # find wav files in breaks folder
    wav_files = glob.glob(f"{samples_dir}/*.wav")
    if not wav_files:
        print(f"no wav files found in {samples_dir} folder!")
        print("please add breakbeat wav files to the breaks folder")
        return False
    
    print(f"found {len(wav_files)} wav files in breaks folder")
    
    try:
        from main import GenBreaks
        
        print("initializing neural audio generator...")
        genbreaks = GenBreaks(mode="audio", sample_length=44100)
        
        print(f"training on {len(wav_files)} breakbeat samples (500 epochs)...")
        genbreaks.train_on_directory(samples_dir, epochs=500, batch_size=16)
        
        print("generating 3 new breakbeat samples...")
        samples = genbreaks.generate_multiple_samples(num_samples=3, output_dir="generated_breaks")
        
        print("saving trained model...")
        genbreaks.save_model("trained_model.pth")
        
        print("\n++ audio demo completed! ++\n")
        print("generated files:")
        
        generated_files = glob.glob("generated_breaks/*.wav")
        for file in generated_files:
            print(f"  - {file}")
        print("  - trained_model.pth")
        
        return True
        
    except ImportError as e:
        print(f"error: {e}")
        print("please install dependencies: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"error: {e}")
        return False


def run_sequence_demo():
    print("++ genbreaks sequence generation demo ++\n")
    
    # check for samples in breaks folder
    samples_dir = "breaks"
    if not os.path.exists(samples_dir):
        print(f"error: {samples_dir} folder not found!")
        print("please create a breaks folder with breakbeat wav files")
        return False
    
    # find wav files in breaks folder
    wav_files = glob.glob(f"{samples_dir}/*.wav")
    if not wav_files:
        print(f"no wav files found in {samples_dir} folder!")
        print("please add breakbeat wav files to the breaks folder")
        return False
    
    # use first file or let user choose
    if len(wav_files) == 1:
        breakbeat_file = wav_files[0]
    else:
        print("choose a breakbeat file:")
        for i, file in enumerate(wav_files, 1):
            print(f"  {i}. {os.path.basename(file)}")
        try:
            choice = int(input(f"choose file (1-{len(wav_files)}): ")) - 1
            breakbeat_file = wav_files[choice if 0 <= choice < len(wav_files) else 0]
        except (ValueError, KeyboardInterrupt):
            breakbeat_file = wav_files[0]
    
    try:
        from main import GenBreaks
        
        print("initializing sequence generator...")
        genbreaks = GenBreaks(mode="sequence")
        
        print("loading and slicing breakbeat...")
        try:
            num_slices = int(input("enter number of slices (default 16): ") or "16")
        except (ValueError, KeyboardInterrupt):
            num_slices = 16
        genbreaks.load_breakbeat(breakbeat_file, num_slices=num_slices)
        
        # debug: show slice information
        print(f"loaded {len(genbreaks.slices)} slices")
        total_duration = sum(len(slice_audio) for slice_audio in genbreaks.slices) / 44100
        print(f"total slice duration: {total_duration:.3f} seconds")
        
        # debug: show individual slice durations
        print("individual slice durations:")
        for i, slice_audio in enumerate(genbreaks.slices):
            duration = len(slice_audio) / 44100
            print(f"  slice {i+1}: {duration:.3f}s")
        
        print("generating 3 beat sequences...")
        sequences = []
        for i in range(3):
            sequence = genbreaks.generate_sequence(temperature=0.5)
            sequences.append(sequence)
            print(f"sequence {i+1}: {sequence}")
        
        print("rendering sequences to wav files...")
        base_name = os.path.splitext(os.path.basename(breakbeat_file))[0]
        timestamp = int(time.time())
        
        generated_files = []
        for i, seq in enumerate(sequences, 1):
            output_file = f"sliced_breaks/generated_{base_name}_{timestamp}_{i}.wav"
            os.makedirs("sliced_breaks", exist_ok=True)
            
            # debug: calculate expected duration based on sequence
            expected_duration = sum(len(genbreaks.slices[slice_num-1]) for slice_num in seq) / 44100
            print(f"sequence {i} expected duration: {expected_duration:.3f}s (slices: {seq})")
            
            rendered_audio = genbreaks.render_sequence(seq, output_file, volume=0.8)
            generated_duration = len(rendered_audio) / 44100
            print(f"sequence {i} actual duration: {generated_duration:.3f} seconds")
            generated_files.append(output_file)
        
        print("\n++ sequence demo completed! ++\n")
        print("generated files:")
        for file in generated_files:
            print(f"  - {file}")
        
        return True
        
    except ImportError as e:
        print(f"error: {e}")
        print("please install dependencies: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"error: {e}")
        return False


def run_demo():
    print("++ genbreaks demo ++\n")
    print("modes:")
    print("1. audio generation: neural network generates breakbeat audio from scratch")
    print("2. sequence generation: generates musical sequences from breakbeat slice sequences\n")
    
    try:
        choice = input("choose demo mode (1=audio, 2=sequence, 3=both): ").strip()
        
        if choice == "1":
            return run_audio_demo()
        elif choice == "2":
            return run_sequence_demo()
        elif choice == "3":
            print("\n" + "="*50)
            success1 = run_audio_demo()
            print("\n" + "="*50)
            success2 = run_sequence_demo()
            return success1 and success2
        else:
            print("invalid choice, exiting")
            return False
            
    except KeyboardInterrupt:
        print("\ndemo cancelled by user")
        return False


if __name__ == "__main__":
    run_demo()