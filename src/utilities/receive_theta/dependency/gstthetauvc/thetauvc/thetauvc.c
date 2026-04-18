/*
 * Copyright 2020-2021 K. Takeo. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials provided
 * with the distribution.
 * 3. Neither the name of the author nor other contributors may be
 * used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "libuvc/libuvc.h"
#include "thetauvc.h"

#define USBVID_RICOH 0x05ca
#define USBPID_THETAV_UVC 0x2712
#define USBPID_THETAZ1_UVC 0x2715

struct thetauvc_mode
{
    unsigned int mode;
    unsigned int width;
    unsigned int height;
    unsigned int fps;
};

typedef struct thetauvc_mode thetauvc_mode_t;

static thetauvc_mode_t stream_mode[] = {
    {
     .mode = THETAUVC_MODE_UHD_2997,
     .width = 3840,
     .height = 1920,
     .fps = 29,
     },
    {
     .mode = THETAUVC_MODE_FHD_2997,
     .width = 1920,
     .height = 960,
     .fps = 29,
     },
    {
     .mode = THETAUVC_MODE_NUM,
     .width = 0,
     .height = 0,
     .fps = 0,
     }
};

uvc_error_t
thetauvc_find_devices(uvc_context_t * ctx, uvc_device_t *** devs)
{
    uvc_device_t **devlist, **ret;
    uvc_device_t *dev;
    uvc_error_t res;

    int     idx, devcnt;

    res = uvc_find_devices(ctx, &devlist, USBVID_RICOH, 0, NULL);
    if (res != UVC_SUCCESS) {
	return res;
    }

    ret = (uvc_device_t **) malloc(sizeof(uvc_device_t *));
    if (ret == NULL) {
	uvc_free_device_list(devlist, 1);
	return UVC_ERROR_NO_MEM;
    }
    *ret = NULL;

    for (idx = 0, devcnt = 0; (dev = devlist[idx]) != NULL; idx++) {
	uvc_device_descriptor_t *desc;

	if (uvc_get_device_descriptor(dev, &desc) != UVC_SUCCESS)
	    continue;

	if (desc->idProduct == USBPID_THETAV_UVC
	    || desc->idProduct == USBPID_THETAZ1_UVC) {
	    void   *tmp_ptr;

	    devcnt++;

	    tmp_ptr = realloc(ret, (devcnt + 1) * sizeof(uvc_device_t *));
	    if (tmp_ptr == NULL) {
		uvc_free_device_list(devlist, 1);
		uvc_free_device_descriptor(desc);
		free(ret);
		return UVC_ERROR_NO_MEM;
	    }

	    ret = tmp_ptr;
	    ret[devcnt - 1] = dev;
	    ret[devcnt] = NULL;
	}
	uvc_free_device_descriptor(desc);
    }

    for (idx = 0; idx < devcnt; idx++)
	uvc_ref_device(ret[idx]);

    uvc_free_device_list(devlist, 1);

    if (devcnt) {
	*devs = ret;
	return UVC_SUCCESS;
    } else {
	free(ret);
	return UVC_ERROR_NO_DEVICE;
    }

}

uvc_error_t
thetauvc_print_devices(uvc_context_t * ctx, FILE * fp)
{
    uvc_device_t **devlist;
    uvc_error_t res;
    FILE   *outfp;
    int     idx;

    outfp = (fp == NULL) ? stdout : fp;
    res = thetauvc_find_devices(ctx, &devlist);
    if (res != UVC_SUCCESS) {
	uvc_perror(res, "");
	return res;
    }

    fprintf(outfp, "No : %-18s : %-10s\n", "Product", "Serial");
    for (idx = 0; devlist[idx] != NULL; idx++) {
	uvc_device_descriptor_t *desc;

	if (uvc_get_device_descriptor(devlist[idx], &desc) != UVC_SUCCESS)
	    continue;

	fprintf(outfp, "%2d : %-18s : %-10s\n", idx, desc->product,
		desc->serialNumber);
	uvc_free_device_descriptor(desc);
    }

    uvc_free_device_list(devlist, 1);

    return UVC_SUCCESS;
}

uvc_error_t
thetauvc_find_device(uvc_context_t * ctx, uvc_device_t ** devh,
		     unsigned int index)
{
    uvc_device_t **devlist;
    uvc_error_t res;
    unsigned int idx;

    res = thetauvc_find_devices(ctx, &devlist);
    if (res != UVC_SUCCESS)
	return res;

    for (idx = 0; idx <= index; idx++) {
	if (devlist[idx] == NULL) {
	    uvc_free_device_list(devlist, 1);
	    return UVC_ERROR_NO_DEVICE;
	}
    }

    uvc_ref_device(devlist[index]);
    *devh = devlist[index];
    uvc_free_device_list(devlist, 1);

    return UVC_SUCCESS;
}

uvc_error_t
thetauvc_find_device_by_serial(uvc_context_t * ctx, uvc_device_t ** devh,
			       const char *serial)
{
    uvc_device_t **devlist, *dev;
    uvc_error_t res;
    unsigned int idx;
    int     found;

    res = thetauvc_find_devices(ctx, &devlist);
    if (res != UVC_SUCCESS)
	return res;

    found = 0;
    if (serial == NULL) {
	dev = devlist[0];
	found = 1;
    } else {
	uvc_device_descriptor_t *desc;

	idx = 0;
	do {
	    dev = devlist[idx++];
	    if (dev == NULL) {
		res = UVC_ERROR_NO_DEVICE;
		break;
	    }

	    if (uvc_get_device_descriptor(dev, &desc) != UVC_SUCCESS)
		continue;

	    if (desc->serialNumber && !strcmp(desc->serialNumber, serial))
		found = 1;

	    uvc_free_device_descriptor(desc);

	} while (found != 1);
    }

    if (dev != NULL) {
	uvc_ref_device(dev);
	*devh = dev;
    }

    uvc_free_device_list(devlist, 1);
    return res;
}

uvc_error_t
thetauvc_get_stream_ctrl_format_size(uvc_device_handle_t * devh,
				     unsigned int mode,
				     uvc_stream_ctrl_t * ctrl)
{
    uvc_error_t res;
    thetauvc_mode_t *m;
    int     i;

    if (!(mode < THETAUVC_MODE_NUM))
	return UVC_ERROR_INVALID_MODE;

    for (i = 0; i < THETAUVC_MODE_NUM; i++) {
	if (stream_mode[i].mode == mode)
	    m = &stream_mode[i];
    }

    res = uvc_get_stream_ctrl_format_size(devh, ctrl,
					  UVC_FRAME_FORMAT_H264, m->width,
					  m->height, m->fps);

    return res;
}

uvc_error_t
thetauvc_run_streaming(uvc_device_t * dev, uvc_device_handle_t ** devh,
		       unsigned int mode, uvc_frame_callback_t * cb,
		       void *pdata)
{
    uvc_error_t res;
    uvc_stream_ctrl_t ctrl;

    res = uvc_open(dev, devh);
    if (res != UVC_SUCCESS)
	return res;
    thetauvc_get_stream_ctrl_format_size(*devh, mode, &ctrl);
    if (res != UVC_SUCCESS)
	printf("error");

    res = uvc_start_streaming(*devh, &ctrl, cb, pdata, 0);

    uvc_close(*devh);

    return UVC_SUCCESS;
}
