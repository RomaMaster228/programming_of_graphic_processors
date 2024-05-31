#include <stdio.h>
#include <math.h>
#include <string.h>


typedef unsigned char uchar;


struct uchar4 {
	uchar x;
	uchar y;
	uchar z;
	uchar w;
};


struct vec3 {
    double x;
    double y;
    double z;
};


struct uint4 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
};


typedef struct vec3 vec3;
typedef struct uchar4 uchar4;
typedef struct uint4 uint4;


double dot(vec3 a, vec3 b) {
	  return a.x * b.x + a.y * b.y + a.z * b.z;
}

/*
| i  j  k  |
| x1 y1 z1 | = i * (y1 * z2 - z1 * y2) + j * ... + k * ...
| x2 y2 z2 |
*/


vec3 prod(vec3 a, vec3 b) {
	  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,  a.x * b.y - a.y * b.x};
}


vec3 norm(vec3 v) {
	  double l = sqrt(dot(v, v));
	  return {v.x / l, v.y / l, v.z / l};
}


vec3 diff(vec3 a, vec3 b) {
	  return {a.x - b.x, a.y - b.y, a.z - b.z};
}


vec3 add(vec3 a, vec3 b) {
	  return {a.x + b.x, a.y + b.y, a.z + b.z};
}


vec3 mult(vec3 a, vec3 b, vec3 c, vec3 v) {
	  return {a.x * v.x + b.x * v.y + c.x * v.z,
			      a.y * v.x + b.y * v.y + c.y * v.z,
			      a.z * v.x + b.z * v.y + c.z * v.z};
}

vec3 mult_num(vec3 a, double num) {
	  return {a.x * num, a.y * num, a.z * num};
}


struct trig {
    vec3 a;
    vec3 b;
    vec3 c;
    uchar4 color;
};


uchar4 ray(vec3 pos, vec3 dir, vec3 light_pos, uchar4 light_color, trig* trigs, int trigs_cnt) {
    int k_min = -1;
    double ts_min;
    for (int k = 0; k < trigs_cnt; k++) {
        vec3 e1 = diff(trigs[k].b, trigs[k].a);
        vec3 e2 = diff(trigs[k].c, trigs[k].a);
        vec3 p = prod(dir, e2);
        double div = dot(p, e1);
        if (fabs(div) < 1e-10) {
          continue;
        }
        vec3 t = diff(pos, trigs[k].a);
        double u = dot(p, t) / div;
        if (u < 0.0 || u > 1.0) {
          continue;
        }
        vec3 q = prod(t, e1);
        double v = dot(q, dir) / div;
        if (v < 0.0 || v + u > 1.0) {
          continue;
        }
        double ts = dot(q, e2) / div;
        if (ts < 0.0) {
          continue;
        }
        if (k_min == -1 || ts < ts_min) {
          k_min = k;
          ts_min = ts;
        }
    }

    if (k_min == -1) {
      return {0, 0, 0, 255};
    }

    vec3 new_pos = add(mult_num(dir, ts_min), pos);
    vec3 new_dir = diff(light_pos, new_pos);
    double length = sqrt(dot(new_dir, new_dir));
    new_dir = norm(new_dir);

    for (int k = 0; k < trigs_cnt; k++) {
        vec3 e1 = diff(trigs[k].b, trigs[k].a);
        vec3 e2 = diff(trigs[k].c, trigs[k].a);
        vec3 p = prod(new_dir, e2);
        double div = dot(p, e1);
        if (fabs(div) < 1e-10) {
            continue;
        }
        vec3 t = diff(new_pos, trigs[k].a);
        double u = dot(p, t) / div;
        if (u < 0.0 || u > 1.0) {
            continue;
        }
        vec3 q = prod(t, e1);
        double v = dot(q, new_dir) / div;
        if (v < 0.0 || v + u > 1.0) {
            continue;
        }
        double ts = dot(q, e2) / div;
        if (ts > 0.0 && ts < length && k != k_min) {
            return {0, 0, 0, 255};
        }
    }

    return {uchar(trigs[k_min].color.x * light_color.x), uchar(trigs[k_min].color.y * light_color.y), uchar(trigs[k_min].color.z * light_color.z), 255};
}

void render(vec3 pc, vec3 pv, int w, int h, double angle, uchar4 *data, vec3 light_pos, uchar4 light_color, trig* trigs, int trigs_cnt) {
    double dw = 2.0 / (w - 1.0);
    double dh = 2.0 / (h - 1.0);
    double z = 1.0 / tan(angle * M_PI / 360.0);
    vec3 bz = norm(diff(pv, pc));
    vec3 bx = norm(prod(bz, {0.0, 0.0, 1.0}));
    vec3 by = norm(prod(bx, bz));
    for(int i = 0; i < w; i++) {
        for(int j = 0; j < h; j++) {
          vec3 v = {-1.0 + dw * i, (-1.0 + dh * j) * h / w, z};
          vec3 dir = mult(bx, by, bz, v);
          data[(h - 1 - j) * w + i] = ray(pc, norm(dir), light_pos, light_color, trigs, trigs_cnt);
        }
    }
}


void ssaa(uchar4* data, uchar4* ssaa_data, int w, int h, int sqrt_rays) {
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            uint4 tmp = {0, 0, 0, 0};
            for (int i = 0; i < sqrt_rays; ++i) {
                for (int j = 0; j < sqrt_rays; ++j) {
                    uchar4 cur_pixel = data[w * sqrt_rays * (y * sqrt_rays + j) + (x * sqrt_rays + i)];
                    tmp.x += cur_pixel.x;
                    tmp.y += cur_pixel.y;
                    tmp.z += cur_pixel.z;
                }
            }
            int rays_per_pixel = sqrt_rays * sqrt_rays;
            ssaa_data[y * w + x] = (uchar4){uchar(tmp.x / rays_per_pixel), uchar(tmp.y / rays_per_pixel), uchar(tmp.z / rays_per_pixel), 255};
        }
    }
}


void make_floor(vec3 a, vec3 b, vec3 c, vec3 d, uchar4 color, trig* trigs, int i) {
    trigs[i] = {a, b, c, color};
    trigs[i + 1] = {a, c, d, color};
}


void make_tetrahedron(vec3 center, uchar4 color, double r, trig* trigs, int i) {
    double a = r * sqrt(3);
    vec3 vertices[] = {{center.x - a / 2, 0, center.z - a / sqrt(12)}, {center.x, center.y + r, center.z - a / sqrt(12)}, {center.x + a / 2, 0, center.z - a / sqrt(12)}, {center.x, center.y, center.z + r}};
    trigs[i] = {vertices[0], vertices[1], vertices[2], color};
    trigs[i + 1] = {vertices[0], vertices[1], vertices[3], color};
    trigs[i + 2] = {vertices[0], vertices[2], vertices[3], color};
    trigs[i + 3] = {vertices[1], vertices[2], vertices[3], color};
}


void make_hexahedron(vec3 center, uchar4 color, double r, trig* trigs, int i) {
    double a = 2 * r / sqrt(3);
    vec3 first_v = {center.x - a / 2, center.y - a / 2, center.z - a / 2};
    vec3 vertices[] = {
        {first_v.x, first_v.y, first_v.z},
        {first_v.x, first_v.y + a, first_v.z},
        {first_v.x + a, first_v.y + a, first_v.z},
        {first_v.x + a, first_v.y, first_v.z},
        {first_v.x, first_v.y, first_v.z + a},
        {first_v.x, first_v.y + a, first_v.z + a},
        {first_v.x + a, first_v.y + a, first_v.z + a},
        {first_v.x + a, first_v.y, first_v.z + a}
    };
    trigs[i] = {vertices[0], vertices[1], vertices[2], color};
    trigs[i + 1] = {vertices[2], vertices[3], vertices[0], color};
    trigs[i + 2] = {vertices[6], vertices[7], vertices[3], color};
    trigs[i + 3] = {vertices[3], vertices[2], vertices[6], color};
    trigs[i + 4] = {vertices[2], vertices[1], vertices[5], color};
    trigs[i + 5] = {vertices[5], vertices[6], vertices[2], color};
    trigs[i + 6] = {vertices[4], vertices[5], vertices[1], color};
    trigs[i + 7] = {vertices[1], vertices[0], vertices[4], color};
    trigs[i + 8] = {vertices[3], vertices[7], vertices[4], color};
    trigs[i + 9] = {vertices[4], vertices[0], vertices[3], color};
    trigs[i + 10] = {vertices[6], vertices[5], vertices[4], color};
    trigs[i + 11] = {vertices[4], vertices[7], vertices[6], color};
}


void make_dodecahedron(vec3 center, uchar4 color, double r, trig* trigs, int i) {
    double a = (1 + sqrt(5)) / 2;
    double b = 1 / a;
    vec3 vertices[] = {
        {-b, 0, a},
        {b, 0, a},
        {-1, 1, 1},
        {1, 1, 1},
        {1, -1, 1},
        {-1, -1, 1},
        {0, -a, b},
        {0, a, b},
        {-a, -b, 0},
        {-a, b, 0},
        {a, b, 0},
        {a, -b, 0},
        {0, -a, -b},
        {0, a, -b},
        {1, 1, -1},
        {1, -1, -1},
        {-1, -1, -1},
        {-1, 1, -1},
        {b, 0, -a},
        {-b, 0, -a},
    };
    for (int i = 0; i < 20; i++) {
        vertices[i].x = vertices[i].x * r / sqrt(3) + center.x;
        vertices[i].y = vertices[i].y * r / sqrt(3) + center.y;
        vertices[i].z = vertices[i].z * r / sqrt(3) + center.z;
    }
    trigs[i] = {vertices[4], vertices[0], vertices[6], color};
    trigs[i + 1] = {vertices[0], vertices[5], vertices[6], color};
    trigs[i + 2] = {vertices[0], vertices[4], vertices[1], color};
    trigs[i + 3] = {vertices[0], vertices[3], vertices[7], color};
    trigs[i + 4] = {vertices[2], vertices[0], vertices[7], color};
    trigs[i + 5] = {vertices[0], vertices[1], vertices[3], color};
    trigs[i + 6] = {vertices[10], vertices[1], vertices[11], color};
    trigs[i + 7] = {vertices[3], vertices[1], vertices[10], color};
    trigs[i + 8] = {vertices[1], vertices[4], vertices[11], color};
    trigs[i + 9] = {vertices[5], vertices[0], vertices[8], color};
    trigs[i + 10] = {vertices[0], vertices[2], vertices[9], color};
    trigs[i + 11] = {vertices[8], vertices[0], vertices[9], color};
    trigs[i + 12] = {vertices[5], vertices[8], vertices[16], color};
    trigs[i + 13] = {vertices[6], vertices[5], vertices[12], color};
    trigs[i + 14] = {vertices[12], vertices[5], vertices[16], color};
    trigs[i + 15] = {vertices[4], vertices[12], vertices[15], color};
    trigs[i + 16] = {vertices[4], vertices[6], vertices[12], color};
    trigs[i + 17] = {vertices[11], vertices[4], vertices[15], color};
    trigs[i + 18] = {vertices[2], vertices[13], vertices[17], color};
    trigs[i + 19] = {vertices[2], vertices[7], vertices[13], color};
    trigs[i + 20] = {vertices[9], vertices[2], vertices[17], color};
    trigs[i + 21] = {vertices[13], vertices[3], vertices[14], color};
    trigs[i + 22] = {vertices[7], vertices[3], vertices[13], color};
    trigs[i + 23] = {vertices[3], vertices[10], vertices[14], color};
    trigs[i + 24] = {vertices[8], vertices[17], vertices[19], color};
    trigs[i + 25] = {vertices[16], vertices[8], vertices[19], color};
    trigs[i + 26] = {vertices[8], vertices[9], vertices[17], color};
    trigs[i + 27] = {vertices[14], vertices[11], vertices[18], color};
    trigs[i + 28] = {vertices[11], vertices[15], vertices[18], color};
    trigs[i + 29] = {vertices[10], vertices[11], vertices[14], color};
    trigs[i + 30] = {vertices[12], vertices[19], vertices[18], color};
    trigs[i + 31] = {vertices[15], vertices[12], vertices[18], color};
    trigs[i + 32] = {vertices[12], vertices[16], vertices[19], color};
    trigs[i + 33] = {vertices[19], vertices[13], vertices[18], color};
    trigs[i + 34] = {vertices[17], vertices[13], vertices[19], color};
    trigs[i + 35] = {vertices[13], vertices[14], vertices[18], color};
}


void print_default() {
    printf("120\n");
    printf("res/%%d.data\n");
    printf("640 480 120\n");
    printf("7.0 3.0 0.0 2.0 1.0 2.0 6.0 1.0 0.0 0.0\n");
    printf("2.0 0.0 0.0 0.5 0.1 1.0 4.0 1.0 0.0 0.0\n");
    printf("3.0 2.0 0.6 0.3 0.75 0.0 1.0\n");
    printf("0.0 0.0 0.0 0.6 0.25 0.55 1.75\n");
    printf("-3.0 -2.0 0.0 0.0 0.8 0.7 1.5\n");
    printf("-5.0 -5.0 -1.0 -5.0 5.0 -1.0 5.0 5.0 -1.0 5.0 -5.0 -1.0 1.0 0.9 0.35\n");
    printf("-10.0 0.0 12.0 0.4 0.3 0.1\n");
    printf("4\n");
}


void print_vec3(vec3 v) {
    printf("[%lf, %lf, %lf], ", v.x, v.y, v.z);
}


void print_trig(trig t) {
    print_vec3(t.a);
    print_vec3(t.b);
    print_vec3(t.c);
} 


int main(int argc, char* argv[]) {
    bool gpu = true;
    if (argc >= 2) {
        if (strcmp(argv[1], "--default") == 0) {
            print_default();
            return 0;
        }
        if (strcmp(argv[1], "--cpu") == 0) {
            gpu = false;
        }
    }

    // ввод данных по заданию
    int frames_number;
    char output_path[256];
    int w, h;
    double angle;
    double r0c, z0c, phi0c, Arc, Azc, wrc, wzc, wphic, prc, pzc;
    double r0n, z0n, phi0n, Arn, Azn, wrn, wzn, wphin, prn, pzn;
    double center1_x, center1_y, center1_z, color1_x, color1_y, color1_z, r1;
    double center2_x, center2_y, center2_z, color2_x, color2_y, color2_z, r2;
    double center3_x, center3_y, center3_z, color3_x, color3_y, color3_z, r3;
    double floor1_x, floor1_y, floor1_z, floor2_x, floor2_y, floor2_z, floor3_x, floor3_y, floor3_z, floor4_x, floor4_y, floor4_z, floor_color_x, floor_color_y, floor_color_z;
    double light_pos_x, light_pos_y, light_pos_z;
    double light_color_x, light_color_y, light_color_z;
    double sqrt_rays;
    scanf("%d", &frames_number);
    scanf("%s", output_path);
    scanf("%d%d%lf", &w, &h, &angle);
    scanf("%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf", &r0c, &z0c, &phi0c, &Arc, &Azc, &wrc, &wzc, &wphic, &prc, &pzc);
    scanf("%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf", &r0n, &z0n, &phi0n, &Arn, &Azn, &wrn, &wzn, &wphin, &prn, &pzn);
    scanf("%lf%lf%lf%lf%lf%lf%lf", &center1_x, &center1_y, &center1_z, &color1_x, &color1_y, &color1_z, &r1);
    scanf("%lf%lf%lf%lf%lf%lf%lf", &center2_x, &center2_y, &center2_z, &color2_x, &color2_y, &color2_z, &r2);
    scanf("%lf%lf%lf%lf%lf%lf%lf", &center3_x, &center3_y, &center3_z, &color3_x, &color3_y, &color3_z, &r3);
    scanf("%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf", &floor1_x, &floor1_y, &floor1_z, &floor2_x, &floor2_y, &floor2_z, &floor3_x, &floor3_y, &floor3_z, &floor4_x, &floor4_y, &floor4_z, &floor_color_x, &floor_color_y, &floor_color_z);
    scanf("%lf%lf%lf", &light_pos_x, &light_pos_y, &light_pos_z);
    scanf("%lf%lf%lf", &light_color_x, &light_color_y, &light_color_z);
    scanf("%lf", &sqrt_rays);

    vec3 light_pos = (vec3){light_pos_x, light_pos_y, light_pos_z};
    uchar4 light_color = (uchar4){uchar(light_color_x * 255), uchar(light_color_y * 255), uchar(light_color_z * 255), 255};
    
    uchar4* data = (uchar4*)malloc(sizeof(uchar4) * w * h * sqrt_rays * sqrt_rays);
    uchar4* ssaa_data = (uchar4*)malloc(sizeof(uchar4) * w * h);
    uchar4* dev_data;
    uchar4* dev_ssaa_data;
    trig* dev_trigs;
    char buffer[256];
    trig trigs[54];
    make_floor((vec3){floor1_x, floor1_y, floor1_z}, (vec3){floor2_x, floor2_y, floor2_z}, (vec3){floor3_x, floor3_y, floor3_z}, (vec3){floor4_x, floor4_y, floor4_z}, (uchar4){uchar(floor_color_x * 255), uchar(floor_color_y * 255), uchar(floor_color_z * 255), 255}, trigs, 0);
    make_tetrahedron((vec3){center1_x, center1_y, center1_z}, (uchar4){uchar(color1_x * 255), uchar(color1_y * 255), uchar(color1_z * 255), 255}, r1, trigs, 2);
    make_hexahedron((vec3){center2_x, center2_y, center2_z}, (uchar4){uchar(color2_x * 255), uchar(color2_y * 255), uchar(color2_z * 255), 255}, r2, trigs, 6);
    make_dodecahedron((vec3){center3_x, center3_y, center3_z}, (uchar4){uchar(color3_x * 255), uchar(color3_y * 255), uchar(color3_z * 255), 255}, r3, trigs, 18);
    printf("[");
    for (int i = 0; i < 2; i++) {
    	print_trig(trigs[i]);	
    }
    printf("]\n");
    printf("[");
    for (int i = 2; i < 6; i++) {
    	print_trig(trigs[i]);	
    }
    printf("]\n");
    printf("[");
    for (int i = 6; i < 18; i++) {
    	print_trig(trigs[i]);	
    }
    printf("]\n");
    printf("[");
    for (int i = 18; i < 54; i++) {
    	print_trig(trigs[i]);	
    }
    printf("]\n");

    printf("[");
    for (int frame = 0; frame < frames_number; frame++) {
        double t = 2 * M_PI * frame / frames_number;
        vec3 pc, pv;
        double rc = r0c + Arc * sin(wrc * t + prc);
        double zc = z0c + Azc * sin(wzc * t + pzc);
        double phic = phi0c + wphic * t;
        double rn = r0n + Arn * sin(wrn * t + prn);
        double zn = z0n + Azn * sin(wzn * t + pzn);
        double phin = phi0n + wphin * t;
        pc.x = rc * cos(phic);
        pc.y = rc * sin(phic);
        pc.z = zc;
        pv.x = rn * cos(phin);
        pv.y = rn * sin(phin);
        pv.z = zn;
        print_vec3(pc);
        print_vec3(pv);


        render(pc, pv, w * sqrt_rays, h * sqrt_rays, angle, data, light_pos, light_color, trigs, 54);
        ssaa(data, ssaa_data, w, h, sqrt_rays);

        sprintf(buffer, output_path, frame);
        //printf("%d: %s\n", frame, buffer);
        /*FILE* output_file = fopen(buffer, "wb");
        fwrite(&w, sizeof(int), 1, output_file);
        fwrite(&h, sizeof(int), 1, output_file);
        fwrite(ssaa_data, sizeof(uchar4), w * h, output_file);
        fclose(output_file);*/
    }
    printf("]\n");
    free(data);
    free(ssaa_data);
    return 0;
}
