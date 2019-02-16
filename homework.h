#ifndef HOMEWORK_H1
#define HOMEWORK_H1

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GRAYSCALE '5'
#define COLOR     '6'

#define CHANNELS_NO_GRAY  1
#define CHANNELS_NO_COLOR 3

#define IMG_HEADER_FIELDS 5

#define KERNEL_HEIGHT 3
#define KERNEL_WIDTH  3

#define SMOOTH  "smooth"
#define BLUR    "blur"
#define SHARPEN "sharpen"
#define MEAN    "mean"
#define EMBOSS  "emboss"

#define TAG_INIT  0
#define TAG_LINES 1
#define TAG_END   2

typedef struct {
    unsigned char color[CHANNELS_NO_COLOR];
} pixel;

typedef struct {
    char format;
    // Canale de culoare folosite
    //  1 - doar gri, 3 - RGB
    unsigned char channels;
    unsigned int width;
    unsigned int height;
    unsigned char maxval;
    pixel **matrix;
} image;


/**
 * Aloca memorie pentru o matrice de pixeli de inaltime height si latime
 * width. Matricea este returnata ca parametru.
 **/
void alloc_matrix(pixel ***matrix, int width, int height) {
    *matrix = (pixel**)malloc(height * sizeof(pixel*));

    for (int i = 0; i < height; ++i) {
        (*matrix)[i] = (pixel*)malloc(width * sizeof(pixel));
    }
}

/**
 * Dezaloca memoria unei matrice de pixeli si seteaza pointer-ul pe NULL.
 **/
void free_matrix(pixel ***matrix, int width, int height) {
    for (int i = 0; i < height; ++i) {
        free((*matrix)[i]);
    }
    free(*matrix);
    *matrix = NULL;
}

/**
 * Citeste din fisier matricea de pixeli a imaginii.
 **/
void read_matrix(image *img, FILE *input) {
    unsigned char buffer[img->width * img->channels];

    for (int i = 0; i < img->height; ++i) {
        fread(buffer, sizeof(unsigned char), img->width * img->channels, input);

        for (int j = 0; j < img->width; ++j) {
            for (int k = 0; k < img->channels; ++k) {
                img->matrix[i][j].color[k] = buffer[j * img->channels + k];
            }
        }
    }
}

/**
 * Scrie matricea de pixeli in fisier.
 **/
void write_matrix(image *img, FILE* output) {
    unsigned char buffer[img->width * img->channels];

    for (int i = 0; i < img->height; ++i) {
        for (int j = 0; j < img->width; ++j) {
            for (int k = 0; k < img->channels; ++k) {
                buffer[j * img->channels + k] = img->matrix[i][j].color[k];
            }
        }

        fwrite(buffer, sizeof(unsigned char), img->width * img->channels, output);
    }
}

/**
 * Copiaza pixelii din matricea sursa in matricea destinatie.
 **/
void copy_matrix(pixel **src, pixel **dest, int width, int height,
                 unsigned char channels) {

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            for (int k = 0; k < channels; ++k) {
                dest[i][j].color[k] = src[i][j].color[k];
            }
        }
    }
}

/**
 * Calculeaza culoarea unui pixel folosind formulele prezentate in enunt.
 * Rezultatul este returnat ca parametru.
 **/
void apply_filter_pixel(pixel **matrix, pixel *result, int x, int y,
        const float kernel[KERNEL_HEIGHT][KERNEL_WIDTH],
        unsigned char channels) {

    float mean[channels];
    for (int k = 0; k < channels; ++k) {
        mean[k] = 0.0f;
    }
    for (int i = x - 1; i <= x + 1; ++i) {
        for (int j = y - 1; j <= y + 1; ++j) {
            for (int k = 0; k < channels; ++k) {
                mean[k] += kernel[i - x + 1][j - y + 1]
                         * (float)(matrix[i][j].color[k]);
            }
        }
    }

    for (int k = 0; k < channels; ++k) {
        result->color[k] = (unsigned char)(mean[k]);
    }
}

/**
 * Aplica filtrul pe fiecare pixel din matricea originala si pune rezultatul
 * in matricea copie.
 **/
void apply_filter_serial(image *img_in, image *img_out,
        const float kernel[KERNEL_HEIGHT][KERNEL_WIDTH]) {

    for (int i = 1; i < img_in->height - 1; ++i) {
        for (int j = 1; j < img_in->width - 1; ++j) {
            apply_filter_pixel(img_in->matrix, img_out->matrix[i] + j,
                               i, j, kernel, img_in->channels);
        }
    }
}

#endif /* HOMEWORK_H1 */