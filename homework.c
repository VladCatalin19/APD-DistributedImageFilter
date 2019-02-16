#include <mpi.h>
#include <math.h>
#include "homework.h"

/**
 * Citeste din fisierul de intrare header-ul imaginii si matricea de pixeli
 * a acesteia.
 **/
void read_input(char *file_name, image *img) {
    FILE *input = fopen(file_name, "rb");

    fscanf(input, "P%c\n",   &(img->format));
    fscanf(input, "%d %d\n", &(img->width), &(img->height));
    fscanf(input, "%hhu\n",  &(img->maxval));

    if (img->format == GRAYSCALE) {
        img->channels = CHANNELS_NO_GRAY;
    } else {
        img->channels = CHANNELS_NO_COLOR;
    }

    alloc_matrix(&(img->matrix), img->width, img->height);
    read_matrix(img, input);

    fclose(input);
}

/**
 * Scrie header-ul imaginii si matricea de pixeli in fisierul de iesire.
 **/
void write_data(char *file_name, image *img) {
    FILE *output = fopen(file_name, "wb");

    fprintf(output, "P%c\n",   img->format);
    fprintf(output, "%d %d\n", img->width, img->height);
    fprintf(output, "%hhu\n",  img->maxval);

    write_matrix(img, output);

    fclose(output);
}

/**
 * Copiaza header-ul imaginii originale si aloca memorie pentru o noua matrice
 * cu aceleasi dimensiuni ca cea originala.
 **/
void alloc_image_output(image *img_in, image *img_out) {
    memcpy(img_out, img_in, sizeof(image));
    alloc_matrix(&(img_out->matrix), img_out->width, img_out->height);
}

void copy_image(image *src, image *dest) {
    copy_matrix(src->matrix, dest->matrix, src->width, src->height, src->channels);
}

void free_image(image *img) {
    if(img->matrix != NULL) {
        free_matrix(&(img->matrix), img->width, img->height);
    }
}

int min(int a, int b) {
    return (a < b) ? a : b;
}

int max(int a, int b) {
    return (a > b) ? a : b;
}

/**
 * Structura care retine linia de inceput si final a zonei unde va lucra un proces.
 * Inaltimea este precalculata si retinuta tot aici.
 **/
typedef struct limits_struct {
    int start;
    int end;
    int height;
} LIMITS;

void calculate_line_limits(LIMITS *lim, image *img, int rank, int nProcesses) {
    int start = rank       * img->height / nProcesses;
    int end   = (rank + 1) * img->height / nProcesses;

    lim->start = max(start - 1, 0);
    lim->end   = min(end, img->height - 1);
    lim->height = lim->end - lim->start + 1;
}

/**
 * Ia informatiile din buffer si le pune in matricea pixelilor pe pozitiile lor
 * corecte. Se ia in calcul daca imaginea este alb-negru sau color pentru a calcu-
 * la pozitiile culorilor in matrice.
 **/
void parse_line(image *img, unsigned char *line, int index) {
    for (int j = 0; j < img->width; ++j) {
        for (int k = 0; k < img->channels; ++k) {
            img->matrix[index][j].color[k] = line[j * img->channels + k];
        }
    }
}

/**
 * Aplica filtrele primite ca parametru pe matricea de pixeli a imaginii.
 **/
void apply_filter(char *filter, image *img_in, image *img_out,
    const float smooth_kernel[KERNEL_HEIGHT][KERNEL_WIDTH],
    const float blur_kernel[KERNEL_HEIGHT][KERNEL_WIDTH],
    const float sharpen_kernel[KERNEL_HEIGHT][KERNEL_WIDTH],
    const float mean_kernel[KERNEL_HEIGHT][KERNEL_WIDTH],
    const float emboss_kernel[KERNEL_HEIGHT][KERNEL_WIDTH]) {

    if (strcmp(filter, SMOOTH) == 0) {
        apply_filter_serial(img_in, img_out, smooth_kernel);
    } else if (strcmp(filter, BLUR) == 0) {
        apply_filter_serial(img_in, img_out, blur_kernel);
    } else if (strcmp(filter, SHARPEN) == 0) {
        apply_filter_serial(img_in, img_out, sharpen_kernel);
    } else if (strcmp(filter, MEAN) == 0) {
        apply_filter_serial(img_in, img_out, mean_kernel);
    } else if (strcmp(filter, EMBOSS) == 0) {
        apply_filter_serial(img_in, img_out, emboss_kernel);
    }
}

/**
 * Ia cuclorile pixelilor din matrice si le pune intre-un buffer pentru a fi
 * trimise. Se ia in calcul daca imaginea este alb-negru sau color pentru calcu-
 * larea poziitiei culorii in buffer.
 **/
void create_line_buffer(image *img, unsigned int line, unsigned char *buff) {
    for (int j = 0; j < img->width; ++j) {
        for (int k = 0; k < img->channels; ++k) {
            buff[j * img->channels + k] = img->matrix[line][j].color[k];
        }
    }
}

/**
 * Creeaza un buffer in care pune culorile de pe o linie din matricea de pixeli
 * si o trimite catre destinatie. 
 **/
void send_line(image *img, int line, int rank, int tag) {
    int buff_len = img->width * img->channels;
    unsigned char buff[buff_len];

    create_line_buffer(img, line, buff);

    MPI_Send(buff, buff_len, MPI_UNSIGNED_CHAR, rank, tag, MPI_COMM_WORLD);
}

/**
 * Primeste de la sursa un buffer cu culorile de pe linie si le pune in matricea
 * de pixeli la pozitiile resprective.
 **/
void recv_line(image *img, int line, int rank, int tag) {
    int buff_len = img->width * img->channels;
    unsigned char buff[buff_len];

    MPI_Recv(buff, buff_len, MPI_UNSIGNED_CHAR, rank, tag, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    parse_line(img, buff, line);
}

int main(int argc, char **argv) {
    // Kernel-e predefinite
    const float smooth_kernel[KERNEL_HEIGHT][KERNEL_WIDTH] = {
        {1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f},
        {1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f},
        {1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f}
    };

    const float blur_kernel[KERNEL_HEIGHT][KERNEL_WIDTH] = {
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f},
        {2.0f / 16.0f, 4.0f / 16.0f, 2.0f / 16.0f},
        {1.0f / 16.0f, 2.0f / 16.0f, 1.0f / 16.0f}
    };

    const float sharpen_kernel[KERNEL_HEIGHT][KERNEL_WIDTH] = {
        { 0.0f / 3.0f, -2.0f / 3.0f,  0.0f / 3.0f},
        {-2.0f / 3.0f, 11.0f / 3.0f, -2.0f / 3.0f},
        { 0.0f / 3.0f, -2.0f / 3.0f,  0.0f / 3.0f}
    };

    const float mean_kernel[KERNEL_HEIGHT][KERNEL_WIDTH] = {
        {-1.0f, -1.0f, -1.0f},
        {-1.0f,  9.0f, -1.0f},
        {-1.0f, -1.0f, -1.0f}
    };

    const float emboss_kernel[KERNEL_HEIGHT][KERNEL_WIDTH] = {
        {0.0f,  1.0f, 0.0f},
        {0.0f,  0.0f, 0.0f},
        {0.0f, -1.0f, 0.0f}
    };

    int rank;
    int nProcesses;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

    image img_in;
    image img_out;
    memset(&img_in,  0, sizeof(image));
    memset(&img_out, 0, sizeof(image));

    if (rank == 0) {
        // Citeste imaginea si aloca memorie pentru o copie
        read_input(argv[1], &img_in);
        alloc_image_output(&img_in, &img_out);
        copy_image(&img_in, &img_out);

        if (nProcesses == 1) {
            // Aplica filtrele daca programul se executa serial
            for (int i = 3; i < argc; ++i) {
                apply_filter(argv[i], &img_in, &img_out, smooth_kernel,
                             blur_kernel, sharpen_kernel, mean_kernel,
                             emboss_kernel);


                // Suprascrie imaginea originala cu cea pe care au fost aplicate
                // filtrele doar daca mai trebuie aplicat vreun filtru
                if (i != argc - 1) {
                    copy_image(&img_out, &img_in);
                }
            }
        } else {
            // Trimite header-ul catre celelalte procese
            {
                unsigned int buff[IMG_HEADER_FIELDS];
                buff[0] = img_in.format;
                buff[1] = img_in.channels;
                buff[2] = img_in.width;
                buff[3] = img_in.height;
                buff[4] = img_in.maxval;
                MPI_Bcast(buff, IMG_HEADER_FIELDS, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
            }

            // Calculeaza liniile pe care le va prelucra fiecare proces
            LIMITS limits[nProcesses];
            calculate_line_limits(limits, &img_in, rank, nProcesses);

            // Trimite matricele aferente proceselor dupa rank
            for (int i = 1; i < nProcesses; ++i) {
                calculate_line_limits(limits + i, &img_in, i, nProcesses);

                for (int j = limits[i].start; j <= limits[i].end; ++j) {
                    send_line(&img_in, j, i, TAG_INIT);
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);

            // Creeaza o copie a matricei principale cu dimensiunile calculate
            // anterior
            image img_small_in;
            image img_small_out;

            memcpy(&img_small_in, &img_in, sizeof(image));
            img_small_in.height = limits[0].height;

            memcpy(&img_small_out, &img_out, sizeof(image));
            img_small_out.height = limits[0].height;


            // Aplica filtrele pe bugata lui 0
            for (int i = 3; i < argc; ++i) {
                apply_filter(argv[i], &img_small_in, &img_small_out,
                             smooth_kernel, blur_kernel, sharpen_kernel,
                             mean_kernel, emboss_kernel);

                // Suprascrie imaginea originala cu cea pe care au fost aplicate
                // filtrele doar daca mai trebuie aplicat vreun filtru
                if (i != argc - 1) {
                    copy_image(&img_small_out, &img_small_in);

                    MPI_Barrier(MPI_COMM_WORLD);

                    // Trimite lui 1 penultima linie a lui 0, adica prima linie 
                    // a lui 1
                    send_line(&img_small_in, img_small_in.height - 2, TAG_LINES, TAG_LINES);

                    // Primeste de la 1 a doua linie, adica ultima linie a lui 0
                    recv_line(&img_small_in, img_small_in.height - 1, TAG_LINES, TAG_LINES);
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }

            {
                MPI_Barrier(MPI_COMM_WORLD);

                // Primeste de la toate procesele liniile rezultate
                for (int j = 1; j < nProcesses; ++j) {
                    for (int k = limits[j].start + 1; k < limits[j].end; ++k) {
                        recv_line(&img_out, k, j, TAG_END);
                    }
                }
            }
        }

        write_data(argv[2], &img_out);

    } else {
        // Primeste de la 0 header-ul imaginii
        {
            unsigned int buff[IMG_HEADER_FIELDS];
            MPI_Bcast(buff, IMG_HEADER_FIELDS, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
            img_in.format   = buff[0];
            img_in.channels = buff[1];
            img_in.width    = buff[2];
            img_in.height   = buff[3];
            img_in.maxval   = buff[4];
        }

        // Calculeaza dimensiunea matricei de pixeli care va fi trimisa de 0 si
        // aloca memorie pentru ea
        LIMITS lim;
        calculate_line_limits(&lim, &img_in, rank, nProcesses);

        img_in.height = lim.height;

        memcpy(&img_out, &img_in, sizeof(image));

        alloc_matrix(&(img_in.matrix), img_in.width, img_in.height);

        // Primeste de la 0 liniile
        for (int i = 0; i < lim.height; ++i) {
            recv_line(&img_in, i, 0, TAG_INIT);
        }

        alloc_image_output(&img_in, &img_out);
        copy_image(&img_in, &img_out);

        MPI_Barrier(MPI_COMM_WORLD);

        // Aplica filtrele pentru matricea primita
        for (int i = 3; i < argc; ++i) {
            apply_filter(argv[i], &img_in, &img_out, smooth_kernel,
                         blur_kernel, sharpen_kernel, mean_kernel,
                         emboss_kernel);

            // Suprascrie imaginea originala cu cea pe care au fost aplicate
            // filtrele doar daca mai trebuie aplicat vreun filtru
            if (i != argc - 1) {
                copy_image(&img_out, &img_in);

                MPI_Barrier(MPI_COMM_WORLD);

                // Pentru a evita deadlock unele procese mai intai trimit apoi
                // primesc, iar altele mai intai primesc apoi trimit
                if (rank % 2 == 0) {
                    // Trimite la vecinul de deasupra linia a doua, adica ultima
                    // linie pentru vecin
                    send_line(&img_in, 1, rank - 1, TAG_LINES);

                    // Primeste de la vecinul de deasupraa prima linie, adica
                    // penultima linie pentru vecin
                    recv_line(&img_in, 0, rank - 1, TAG_LINES);

                    if (rank + 1 < nProcesses) {
                        // Trimite la vecinul de dedesubt penultima linie, adica
                        // prima linie pentru vecin
                        send_line(&img_in, img_in.height - 2, rank + 1, TAG_LINES);

                        // Primeste de la vecinul de dedesubt penultima linie,
                        // adica a doua linie pentru vecin
                        recv_line(&img_in, img_in.height - 1, rank + 1, TAG_LINES);
                    }
                } else {
                    // Primeste de la vecinul de deasupraa prima linie, adica
                    // penultima linie pentru vecin
                    recv_line(&img_in, 0, rank - 1, TAG_LINES);

                    // Trimite la vecinul de deasupra linia a doua, adica ultima
                    // linie pentru vecin
                    send_line(&img_in, 1, rank - 1, TAG_LINES);

                    if (rank + 1 < nProcesses) {
                        // Primeste de la vecinul de dedesubt penultima linie,
                        // adica a doua linie pentru vecin
                        recv_line(&img_in, img_in.height - 1, rank + 1, TAG_LINES);

                        // Trimite la vecinul de dedesubt penultima linie, adica
                        // prima linie pentru vecin
                        send_line(&img_in, img_in.height - 2, rank + 1, TAG_LINES);
                    }

                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        {
            MPI_Barrier(MPI_COMM_WORLD);

            // Trimite toate liniile spre 0
            // ssend pentru evitare deadlock
            for (int j = 1; j < lim.height - 1; ++j) {
                int buff_len = img_out.width * img_out.channels;
                unsigned char buff[buff_len];

                create_line_buffer(&img_out, j, buff);

                MPI_Ssend(buff, buff_len, MPI_UNSIGNED_CHAR, 0, 2, MPI_COMM_WORLD);
            }
        }
    }

    free_image(&img_in);
    free_image(&img_out);

    MPI_Finalize();
    return 0;
}