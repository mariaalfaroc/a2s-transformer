import os
import shutil

from music21 import converter as converterm21
from pyMV2H.utils.mv2h import MV2H
from pyMV2H.metrics.mv2h import mv2h
from pyMV2H.utils.music import Music
from pyMV2H.converter.midi_converter import MidiConverter as Converter

from .encoding_convertions import VOICE_CHANGE_TOKEN, STEP_CHANGE_TOKEN


def compute_metrics(y_true, y_pred):
    ################################# Sym-ER and Seq-ER:
    metrics = compute_ed_metrics(y_true=y_true, y_pred=y_pred)
    ################################# MV2H:
    mv2h_dict = compute_mv2h_metrics(y_true=y_true, y_pred=y_pred)
    metrics.update(mv2h_dict)
    return metrics


#################################################################### SYM-ER AND SEQ-ER:


def compute_ed_metrics(y_true, y_pred):
    def levenshtein(a, b):
        n, m = len(a), len(b)

        if n > m:
            a, b = b, a
            n, m = m, n

        current = range(n + 1)
        for i in range(1, m + 1):
            previous, current = current, [i] + [0] * n
            for j in range(1, n + 1):
                add, delete = previous[j] + 1, current[j - 1] + 1
                change = previous[j - 1]
                if a[j - 1] != b[i - 1]:
                    change = change + 1
                current[j] = min(add, delete, change)

        return current[n]

    ed_acc = 0
    length_acc = 0
    label_acc = 0
    for t, h in zip(y_true, y_pred):
        ed = levenshtein(t, h)
        ed_acc += ed
        length_acc += len(t)
        if ed > 0:
            label_acc += 1

    return {
        "sym-er": 100.0 * ed_acc / length_acc,
        "seq-er": 100.0 * label_acc / len(y_pred),
    }


#################################################################### MV2H:


def compute_mv2h_metrics(y_true, y_pred):
    def krn2midi(in_file):
        a = converterm21.parse(in_file).write("midi")
        midi_file = a.name
        shutil.copyfile(a, midi_file)
        os.remove(in_file)
        return midi_file

    def midi2txt(midi_file):
        txt_file = midi_file.replace("mid", "txt")
        converter = Converter(file=midi_file, output=txt_file)
        converter.convert_file()
        with open(txt_file, "r") as fin:
            f = [u.replace(".0", "") for u in fin.readlines()]
        with open(txt_file, "w") as fout:
            for u in f:
                fout.write(u)
        os.remove(midi_file)
        return txt_file

    ########################################### Polyphonic evaluation:

    def eval_as_polyphonic():
        # Convert to MIDI
        reference_midi_file = krn2midi("true.krn")
        predicted_midi_file = krn2midi("pred.krn")

        # Convert to TXT
        reference_txt_file = midi2txt(reference_midi_file)
        predicted_txt_file = midi2txt(predicted_midi_file)

        # Compute MV2H
        reference_file = Music.from_file(reference_txt_file)
        transcription_file = Music.from_file(predicted_txt_file)
        res_dict = MV2H(multi_pitch=0, voice=0, meter=0, harmony=0, note_value=0)
        try:
            res_dict = mv2h(reference_file, transcription_file)
        except:
            pass

        # Remove auxiliar files
        os.remove(reference_txt_file)
        os.remove(predicted_txt_file)

        return res_dict

    ########################################### Monophonic evaluation:

    def get_number_of_voices(kern):
        num_voices = 0
        for token in kern:
            if token == VOICE_CHANGE_TOKEN:
                continue
            if token == STEP_CHANGE_TOKEN:
                break
            num_voices += 1
        return num_voices

    def divide_voice(in_file, out_file, it_voice):
        # Open file
        with open(in_file) as fin:
            read_file = fin.readlines()

        # Read voice
        voice = [u.split("\t")[it_voice].strip() for u in read_file]

        # Write voice
        with open(out_file, "w") as fout:
            for token in voice:
                fout.write(token + "\n")

    def eval_as_monophonic(num_voices):
        global_res_dict = MV2H(multi_pitch=0, voice=0, meter=0, harmony=0, note_value=0)
        for it_voice in range(num_voices):
            # Convert to MIDI
            divide_voice("true.krn", "true_voice.krn", it_voice)
            reference_midi_file = krn2midi("true_voice.krn")
            divide_voice("pred.krn", "pred_voice.krn", it_voice)
            predicted_midi_file = krn2midi("pred_voice.krn")

            # Convert to TXT
            reference_txt_file = midi2txt(reference_midi_file)
            predicted_txt_file = midi2txt(predicted_midi_file)

            # Compute MV2H
            reference_file = Music.from_file(reference_txt_file)
            transcription_file = Music.from_file(predicted_txt_file)
            res_dict = MV2H(multi_pitch=0, voice=0, meter=0, harmony=0, note_value=0)
            try:
                res_dict = mv2h(reference_file, transcription_file)
                global_res_dict.__multi_pitch__ += res_dict.multi_pitch
                global_res_dict.__voice__ += res_dict.voice
                global_res_dict.__meter__ += res_dict.meter
                global_res_dict.__harmony__ += res_dict.harmony
                global_res_dict.__note_value__ += res_dict.note_value
            except:
                pass

            # Remove auxiliar files
            os.remove(reference_txt_file)
            os.remove(predicted_txt_file)

        global_res_dict.__multi_pitch__ /= num_voices
        global_res_dict.__voice__ /= num_voices
        global_res_dict.__meter__ /= num_voices
        global_res_dict.__harmony__ /= num_voices
        global_res_dict.__note_value__ /= num_voices

        return global_res_dict

    ########################################### MV2H evaluation:

    def create_kern_file(out_file, kern, num_voices):
        with open(out_file, "w") as fout:
            # Kern header
            fout.write("\t".join(["**kern"] * num_voices) + "\n")

            # Iterating through the lines
            line = []
            for token in kern:
                if token == STEP_CHANGE_TOKEN:
                    if len(line) > 0:
                        if len(line) < num_voices:
                            line.extend(["."] * (num_voices - len(line)))
                        fout.write("\t".join(line) + "\n")
                    line = []
                else:
                    if token != "DOT" and token != VOICE_CHANGE_TOKEN:
                        line.append(token)
                    else:
                        line.append(".")

    MV2H_global = MV2H(multi_pitch=0, voice=0, meter=0, harmony=0, note_value=0)
    for t, h in zip(y_true, y_pred):
        # Get number of voices
        num_voices = get_number_of_voices(t)

        # GROUND TRUTH
        # Creating ground truth Kern file
        create_kern_file("true.krn", t, num_voices)

        # PREDICTION
        # Creating predicted Kern file
        create_kern_file("pred.krn", h, num_voices)

        # Testing whether predicted Kern can be processed as polyphonic
        flag_polyphonic_kern = True
        try:
            _ = converterm21.parse("pred.krn").write("midi")
        except:
            flag_polyphonic_kern = False

        if flag_polyphonic_kern:
            res_dict = eval_as_polyphonic()
        else:
            res_dict = eval_as_monophonic(num_voices)

        # Updating global results
        MV2H_global.__multi_pitch__ += res_dict.multi_pitch
        MV2H_global.__voice__ += res_dict.voice
        MV2H_global.__meter__ += res_dict.meter
        MV2H_global.__harmony__ += res_dict.harmony
        MV2H_global.__note_value__ += res_dict.note_value

    # Computing average
    MV2H_global.__multi_pitch__ /= len(y_true)
    MV2H_global.__voice__ /= len(y_true)
    MV2H_global.__meter__ /= len(y_true)
    MV2H_global.__harmony__ /= len(y_true)
    MV2H_global.__note_value__ /= len(y_true)

    mv2h_dict = {
        "multi-pitch": MV2H_global.__multi_pitch__,
        "voice": MV2H_global.__voice__,
        "meter": MV2H_global.__meter__,
        "harmony": MV2H_global.__harmony__,
        "note_value": MV2H_global.__note_value__,
        "mv2h": MV2H_global.mv2h,
    }

    return mv2h_dict
