/* GStreamer
 * Copyright (C) 2021 Koji TAKEO <nickel110@icloud.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

#ifndef _GST_THETAUVCSRC_H_
#define _GST_THETAUVCSRC_H_

#include <gst/gst.h>
#include <gst/base/base.h>
#include <gst/video/video.h>

#include "libuvc/libuvc.h"
#include "thetauvc.h"

G_BEGIN_DECLS
#define GST_TYPE_THETAUVCSRC   (gst_thetauvcsrc_get_type())
#define GST_THETAUVCSRC(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_THETAUVCSRC,GstThetauvcsrc))
#define GST_THETAUVCSRC_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_THETAUVCSRC,GstThetauvcsrcClass))
#define GST_IS_THETAUVCSRC(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_THETAUVCSRC))
#define GST_IS_THETAUVCSRC_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_THETAUVCSRC))

typedef struct _GstThetauvcsrc GstThetauvcsrc;
typedef struct _GstThetauvcsrcClass GstThetauvcsrcClass;
typedef enum
{
    GST_THETAUVC_MODE_2K,
    GST_THETAUVC_MODE_4K
} GstThetauvcModeEnum;

GType   gst_thetauvc_mode_get_type(void);

struct _GstThetauvcsrc
{
    GstPushSrc base_thetauvcsrc;

    GMutex  lock;
    GCond   cond;
    GstQueueArray *queue;

    gint    device_number;
    gint    device_index;
    gchar  *serial;
    GstThetauvcModeEnum mode;

    uvc_context_t *ctx;
    uvc_device_t *dev;
    uvc_device_handle_t *devh;
    uvc_stream_ctrl_t ctrl;

    guint64 framecount;
};

struct _GstThetauvcsrcClass
{
    GstPushSrcClass base_thetauvcsrc_class;
};

GType   gst_thetauvcsrc_get_type(void);

G_END_DECLS
#endif
