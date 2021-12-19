import math
import xml.etree.ElementTree as et
import pandas as pd
import numpy as np
import os
import re
import random
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
from sklearn.cluster import KMeans, OPTICS
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical

import persian_normalizer as pn

random.seed(0)

eos_char = "\u0003"


def normalize_location(data: pd.DataFrame):
    xs = data["x"]
    min_x = np.min(xs)
    max_x = np.max(xs)
    width = max_x - min_x
    ys = data["y"]
    min_y = np.min(ys)
    max_y = np.max(ys)
    height = max_y - min_y

    xs -= min_x
    if width != 0:
        xs /= width
    data["x"] = xs
    ys -= min_y
    if height != 0:
        ys /= height
    data["y"] = ys


def extract_feature(data: pd.DataFrame):
    for i in range(len(data)):
        point_type = data.iloc[i, 2]
        if point_type == 1:
            data.iloc[i, [4, 5, 6, 7, 8, 9, 10]] = [0, 0, 0, 1, 0, 1, 0]
            continue

        last_x = data.iloc[i - 1, 0]
        last_y = data.iloc[i - 1, 1]
        x = data.iloc[i, 0]
        y = data.iloc[i, 1]
        # time diff
        time_diff = data.iloc[i, 3] - data.iloc[i - 1, 3]
        # speed
        x_diff = x - last_x
        y_diff = y - last_y
        if time_diff == 0:
            time_diff = 0.00001  # 5 decimals
        x_speed = x_diff / time_diff
        y_speed = y_diff / time_diff
        # direction
        direction_sin = 0
        direction_cos = 1
        diff_norm = math.sqrt(x_diff * x_diff + y_diff * y_diff)
        if diff_norm != 0:
            direction_sin = y_diff / diff_norm
            direction_cos = x_diff / diff_norm
        # angle
        angle_sin = 0
        angle_cos = 1
        if point_type != 2:
            p_prev = [last_x, last_y]
            p_now = [x, y]
            p_next = [data.iloc[i + 1, 0], data.iloc[i + 1, 1]]
            v1 = np.array(p_prev) - np.array(p_now)
            v2 = np.array(p_next) - np.array(p_now)
            angle = np.math.atan2(
                np.linalg.det([v1, v2]), np.dot(v1, v2))
            angle_sin = math.sin(angle)
            angle_cos = math.cos(angle)

        data.iloc[i, [4, 5, 6, 7, 8, 9, 10]] = \
            [x_speed, y_speed, direction_sin, direction_cos, angle_sin, angle_cos, time_diff]


def load_data(dataset_path: str = "data/dataset.pkl.compressed",
              words_dict_path: str = "data/words_dict.pkl.compressed",
              chars_dict_path: str = "data/chars_dict.pkl.compressed",
              truths_file_path: str = "data/truths.pkl.compressed") -> (pd.Series, pd.DataFrame):
    if os.path.exists(dataset_path):
        # load dataset
        dataset: pd.Series = pd.read_pickle(dataset_path, compression="gzip")

        # load text dataset
        dataset_chars: pd.DataFrame = pd.read_pickle(chars_dict_path, compression="gzip")

        print("Data Done!")

        return dataset, dataset_chars

    file_count = valid_writepad_count = line_max_chars = line_max_points = 0

    # load text dataset
    if os.path.exists(chars_dict_path) and os.path.exists(words_dict_path):
        dataset_chars: pd.DataFrame = pd.read_pickle(chars_dict_path, compression="gzip")
        dataset_words: pd.DataFrame = pd.read_pickle(words_dict_path, compression="gzip")

        # set line_max_chars
        words = dataset_words.index
        for w in words:
            if len(w) > line_max_chars:
                line_max_chars = len(w)
    else:
        text_splitter = re.compile(r"^\d+\.\d+\t *(.+?) *$")
        text = ""
        with open("data/HandWritingPhrases.txt", "r", encoding="utf8") as reader:
            line = reader.readline()
            while line != '':
                text += ' ' + text_splitter.match(line).group(1)
                line = reader.readline()
        # chars dictionary
        text = pn.normalize_and_check(text)
        chars_set = ["<unk>", eos_char] + list(set(text))
        chars_one_hot = to_categorical(range(len(chars_set)))
        dataset_chars = pd.DataFrame(chars_one_hot, index=chars_set)
        dataset_chars.to_pickle(chars_dict_path, compression="gzip")
        # words dictionary
        words_set = [s + eos_char for s in list(set(text.split(' ')[1:]))]  # [1:] for skipping first empty entry
        words_chars = []
        for w in words_set:
            chars = []
            for c in w:
                chars.append(dataset_chars.loc[c])
            words_chars.append(chars)
            if len(w) > line_max_chars:
                line_max_chars = len(w)
        # pad words
        pad = dataset_chars.loc["<unk>"]
        pad_len = len(pad)
        for i, w in enumerate(words_chars):
            word_len = len(w)
            if word_len == line_max_chars:
                continue
            padding = np.full((line_max_chars - word_len, pad_len), pad)
            words_chars[i] = list(np.concatenate((w, padding)))
        dataset_words = pd.DataFrame(words_chars, index=words_set)
        dataset_words.to_pickle(words_dict_path, compression="gzip")

    # load truths
    truths = pd.DataFrame({"type": pd.Series(dtype="int32"),
                           "truth_words": pd.Series(dtype=object)})
    if os.path.exists(truths_file_path):
        truths = pd.read_pickle(truths_file_path, compression="gzip")
    else:
        for root, dirs, files in os.walk("data/Dataset/GroundTruths"):
            for filename in files:
                if not filename.startswith("WordGroup"):
                    continue
                file_path = os.path.join(root, filename)
                text_collection = et.parse(file_path).getroot()
                for text in text_collection.findall("text"):
                    truth_id = text.attrib["id"]
                    words = [pn.normalize(w.text) + eos_char for w in text.findall("content/word")]
                    truths.loc[truth_id] = {"type": 1, "truth_words": words}
        truths.to_pickle(truths_file_path, compression="gzip")

    # load points
    xml_namespaces = {"inkml": "http://www.w3.org/2003/InkML"}
    writepads = pd.DataFrame({"points": pd.Series(dtype=object),
                              "truth": pd.Series(dtype=object),
                              "type": pd.Series(dtype="int32")})

    for root, dirs, files in os.walk("data/Dataset/Writepads/WordGroup"):
        for filename in files:
            # if filename != "103.inkml":
            #     continue

            # load file
            file_path = os.path.join(root, filename)
            ink = et.parse(file_path).getroot()
            file_count += 1

            # get truth
            truth_id = ink.find(".//inkml:annotationXML/truthId", xml_namespaces).text
            words = truths.loc[truth_id]["truth_words"]

            # create features dataframe
            drawings = pd.DataFrame({"x": pd.Series(dtype="float32"),
                                     "y": pd.Series(dtype="float32"),
                                     "type": pd.Series(dtype="int32"),
                                     "time": pd.Series(dtype="float32"),
                                     "x_speed": pd.Series(dtype="float32"),
                                     "y_speed": pd.Series(dtype="float32"),
                                     "direction_sin": pd.Series(dtype="float32"),
                                     "direction_cos": pd.Series(dtype="float32"),
                                     "angle_sin": pd.Series(dtype="float32"),
                                     "angle_cos": pd.Series(dtype="float32"),
                                     "time_diff": pd.Series(dtype="float32")})
            stroke_means = pd.Series(dtype="float32")
            real_time = pd.Series(dtype="float32")
            for stroke in ink.findall(".//inkml:trace", xml_namespaces):
                time_offset = float(stroke.attrib["timeOffset"])
                points = stroke.text.split(", ")
                temp_drawings = []
                first_time = 0
                for i, point in enumerate(points):
                    features = point.split(" ")
                    # location
                    x = float(features[0])
                    y = float(features[1])
                    # time
                    time = float(features[3]) + time_offset - first_time
                    if i == 0:
                        first_time = time
                        time = 0

                    temp_drawings.append([x, y, 0, time, 0, 0, 0, 1, 0, 1, time])
                last_drawing_index = len(temp_drawings) - 1
                # starting and ending type
                temp_drawings[0][2] = 1
                temp_drawings[last_drawing_index][2] = 2

                temp_df = pd.DataFrame(temp_drawings, columns=drawings.columns)
                drawings = drawings.append(temp_df, ignore_index=True)

                # calculate stroke means
                temp_s = temp_df["time"] + first_time
                real_time = real_time.append(temp_s, ignore_index=True)
                stroke_means = stroke_means.append(pd.Series(temp_s.mean()), ignore_index=True)

            # normalize
            normalize_location(drawings)

            # line segmentation
            line_number = len(words)

            # init_points, step_size = np.linspace(0, 1, num=line_number, endpoint=False, retstep=True)
            # init_points += step_size / 2
            # km = KMeans(n_clusters=line_number, init=init_points.reshape((-1, 1)), n_init=1, random_state=0)
            km = KMeans(n_clusters=line_number, n_init=20, max_iter=200, random_state=0)

            op = OPTICS(min_samples=2, max_eps=7, xi=0.9)

            def rearrange_line_label(labels: np.ndarray) -> np.ndarray:
                new_labels_mapping = pd.Series(dtype="int32")
                for l in range(line_number):
                    new_labels_mapping.loc[real_time[labels == l].median()] = l
                return np.take(new_labels_mapping.sort_index().argsort().to_numpy(), labels)

            # y based cluster
            points_y_line = km.fit_predict(drawings["y"].to_numpy().reshape((-1, 1)))
            points_y_line = rearrange_line_label(np.asarray(points_y_line))
            drawings["line"] = points_y_line
            # time based cluster
            strokes_time_line = km.fit_predict(stroke_means.to_numpy().reshape((-1, 1)))
            strokes_starting_point_index = drawings[drawings["type"] == 1].index
            # points_time_line = np.zeros((len(points_y_line),), dtype="int32")
            # for i in range(len(strokes_time_line)):
            #     starting_index = strokes_starting_point_index[i]
            #     try:
            #         ending_index = strokes_starting_point_index[i + 1]
            #     except IndexError:
            #         ending_index = len(points_y_line)
            #     points_time_line[starting_index:ending_index] = strokes_time_line[i]
            # points_time_line = rearrange_line_label(points_time_line)
            # set line numbers
            # first pass
            line_column_index = drawings.columns.get_loc("line")
            for i in range(len(strokes_time_line)):
                starting_index = strokes_starting_point_index[i]
                try:
                    ending_index = strokes_starting_point_index[i + 1]
                except IndexError:
                    ending_index = len(points_y_line)
                y_labels_count: pd.Series = pd.Series(points_y_line[starting_index:ending_index]).value_counts()
                line = y_labels_count.idxmax()
                time_line = strokes_time_line[i]
                points_count = y_labels_count.sum()
                if line != time_line \
                        and time_line in y_labels_count.index \
                        and y_labels_count.loc[line] / points_count < 0.75 \
                        and y_labels_count.loc[time_line] / points_count > 0.1:
                    line = time_line
                drawings.iloc[starting_index:ending_index, line_column_index] = line
            # second pass
            for i in range(line_number):
                current = drawings.loc[drawings["line"] == i, ["x", "y"]]
                if len(current) == 0:
                    continue
                points_index = np.asarray(current.index)
                labels = op.fit_predict(points_index.reshape((-1, 1)))
                if len(np.unique(labels)) == 1:
                    continue
                need_check = np.asarray(pd.Series(labels).value_counts(sort=False) < len(labels) / 2).nonzero()[0]
                if i == 0:
                    last = None
                else:
                    last = drawings.loc[drawings["line"] == i - 1, ["x", "y"]]
                current_filtered = current[np.isin(labels, need_check, invert=True)]
                if len(current_filtered) == 0:
                    need_check = np.delete(need_check, np.trunc(len(need_check) / 2).astype('int32'))
                    current_filtered = current[np.isin(labels, need_check, invert=True)]
                boundary = labels[np.isin(labels, need_check, invert=True)].min()
                if i == line_number - 1:
                    next = None
                else:
                    next = drawings.loc[drawings["line"] == i + 1, ["x", "y"]]
                distances = pd.Series(dtype="float32")
                for c in need_check:
                    desired_line = i - 1 if c < boundary else i + 1
                    if desired_line == -1 or desired_line == line_number:
                        desired_line = i
                    loc_means = current[labels == c].mean()
                    if last is not None:
                        distances.loc[i - 1] = ((last - loc_means) ** 2).sum(axis=1).min()
                    distances.loc[i] = ((current_filtered - loc_means) ** 2).sum(axis=1).min()
                    if next is not None:
                        distances.loc[i + 1] = ((next - loc_means) ** 2).sum(axis=1).min()
                    line = distances.sort_values().index[0]
                    if distances.loc[line] == 0:
                        distances.loc[line] = 0.0000001
                    if line != desired_line and distances.loc[desired_line] / distances.loc[line] <= 1.5:
                        line = desired_line
                    if line != i:
                        drawings.loc[points_index[labels == c], "line"] = line

            # add valid drawings
            if len(drawings.iloc[:, line_column_index].unique()) == len(words):
                valid_writepad_count += 1
                save_fig = random.randint(0, 14) == 7
                for line in range(len(words)):
                    points = drawings[drawings["line"] == line].reset_index(drop=True)
                    points.drop("line", axis=1, inplace=True)

                    # normalize line
                    normalize_location(points)

                    # extract features
                    extract_feature(points)

                    points = points.to_numpy()
                    word_chars = np.vstack(dataset_words.loc[words[line]].to_numpy())
                    writepads = writepads.append(
                        pd.DataFrame([[points, word_chars, 1]], columns=writepads.columns),
                        ignore_index=True)
                    if len(points) > line_max_points:
                        line_max_points = len(points)

                    # if save_fig:
                    # plt.scatter(points[:, 0], points[:, 1], s=1.25)
                    # plt.title(get_display(words[line]))
                    # plt.axis([-0.1, 1.1, 1.1, -0.1])
                    # plt.savefig("data/Segmentation/{}_{}.jpg".format(filename, line))
                    # plt.clf()
    # pad points and prepare x and y
    input_features_index = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
    input_features_size = len(input_features_index)
    x = np.empty((0, line_max_points, input_features_size))
    y = np.empty((0, line_max_chars, len(dataset_chars)))
    sum_time_diff = []
    points_count = []
    points_column_index = writepads.columns.get_loc("points")
    truth_column_index = writepads.columns.get_loc("truth")
    for i in range(len(writepads)):
        points = writepads.iloc[i, points_column_index]
        points = points[:, input_features_index]
        points_len = len(points)

        correct_time_diffs = points[points[:, 2] != 1, 9]
        sum_time_diff.append(correct_time_diffs.sum())
        points_count.append(len(correct_time_diffs))

        if points_len < line_max_points:
            padding = np.zeros((line_max_points - points_len, input_features_size))
            points = np.concatenate((points, padding))
        x = np.concatenate((x, [points]))
        y = np.concatenate((y, [writepads.iloc[i, truth_column_index]]))

    # prepare dataset
    x, y = shuffle(x, y, random_state=0)

    # save dataset
    dataset = pd.Series([x, y])
    dataset.to_pickle(dataset_path, compression="gzip")

    # calculate mean time diff
    mean_time_diff = np.sum(sum_time_diff) / np.sum(points_count)

    print("Data Done: {1}/{0}, Line Max Chars: {2}, Line Max Points: {3}, Mean Time Diff: {4}".format(
        file_count, valid_writepad_count, line_max_chars, line_max_points, mean_time_diff))

    return dataset, dataset_chars
