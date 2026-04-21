/*
 * Copyright 2020 K. Takeo. All rights reserved.
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

#if !defined(__THETAUVC_H__)
#define __THETAUVC_H__

#if defined(__cplusplus)
extern "C" {
#endif

enum thetauvc_mode_code {
	THETAUVC_MODE_FHD_2997 = 0,
	THETAUVC_MODE_UHD_2997,
	THETAUVC_MODE_NUM
};

extern uvc_error_t thetauvc_find_devices(uvc_context_t *, uvc_device_t ***);
extern uvc_error_t thetauvc_print_devices(uvc_context_t *, FILE *);
extern uvc_error_t thetauvc_find_device(uvc_context_t *, uvc_device_t **,
	unsigned int);
extern uvc_error_t thetauvc_find_device_by_serial(uvc_context_t *,
	uvc_device_t **, const char *);
extern uvc_error_t thetauvc_get_stream_ctrl_format_size(uvc_device_handle_t *,
	unsigned int, uvc_stream_ctrl_t *);
extern uvc_error_t thetauvc_run_streaming(uvc_device_t *, uvc_device_handle_t **,
	unsigned int, uvc_frame_callback_t *, void *);


#if defined(__cplsplus)
}
#endif
#endif
