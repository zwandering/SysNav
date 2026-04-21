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
 * Free Software Foundation, Inc., 51 Franklin Street, Suite 500,
 * Boston, MA 02110-1335, USA.
 */
/**
 * SECTION:element-gstthetauvcsrc
 *
 * Read live strem from Theta V/Z1.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch-1.0 -v thetauvcsrc ! h264parse ! decodebin ! autovideosink
 * ]|
 * Play live streaming from Theta V/Z1. 
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>

#include <gst/gst.h>

#include "libuvc/libuvc.h"
#include "thetauvc.h"
#include "gstthetauvcsrc.h"

GST_DEBUG_CATEGORY_STATIC(gst_thetauvcsrc_debug_category);
#define GST_CAT_DEFAULT gst_thetauvcsrc_debug_category

const int skipFrameNum = 2;
int skipFrameCount = 0;

/* prototypes */

static void gst_thetauvcsrc_set_property(GObject * object,
    guint property_id, const GValue * value, GParamSpec * pspec);
static void gst_thetauvcsrc_get_property(GObject * object, guint property_id,
    GValue * value, GParamSpec * pspec);
static void gst_thetauvcsrc_dispose(GObject * object);
static void gst_thetauvcsrc_finalize(GObject * object);

static GstCaps *gst_thetauvcsrc_get_caps(GstBaseSrc * src, GstCaps * filter);
static gboolean gst_thetauvcsrc_negotiate(GstBaseSrc * src);
static GstCaps *gst_thetauvcsrc_fixate(GstBaseSrc * src, GstCaps * caps);
static gboolean gst_thetauvcsrc_set_caps(GstBaseSrc * src, GstCaps * caps);
static gboolean gst_thetauvcsrc_decide_allocation(GstBaseSrc * src,
    GstQuery * query);
static gboolean gst_thetauvcsrc_start(GstBaseSrc * src);
static gboolean gst_thetauvcsrc_stop(GstBaseSrc * src);
static void gst_thetauvcsrc_get_times(GstBaseSrc * src, GstBuffer * buffer,
    GstClockTime * start, GstClockTime * end);
static gboolean gst_thetauvcsrc_get_size(GstBaseSrc * src, guint64 * size);
static gboolean gst_thetauvcsrc_is_seekable(GstBaseSrc * src);
static gboolean gst_thetauvcsrc_prepare_seek_segment(GstBaseSrc * src,
    GstEvent * seek, GstSegment * segment);
static gboolean gst_thetauvcsrc_do_seek(GstBaseSrc * src,
    GstSegment * segment);
static gboolean gst_thetauvcsrc_unlock(GstBaseSrc * src);
static gboolean gst_thetauvcsrc_unlock_stop(GstBaseSrc * src);
static gboolean gst_thetauvcsrc_query(GstBaseSrc * src, GstQuery * query);
static gboolean gst_thetauvcsrc_event(GstBaseSrc * src, GstEvent * event);
static GstFlowReturn gst_thetauvcsrc_create(GstPushSrc * src,
    GstBuffer ** buf);
static GstFlowReturn gst_thetauvcsrc_alloc(GstBaseSrc * src, guint64 offset,
    guint size, GstBuffer ** buf);
static GstFlowReturn gst_thetauvcsrc_fill(GstBaseSrc * src, guint64 offset,
    guint size, GstBuffer * buf);

enum
{
    PROP_HW_SERIAL = 1,
    PROP_DEVICE_NUM,
    PROP_MODE,
    PROP_DEVICE_INDEX
};

/* pad templates */

static GstStaticPadTemplate gst_thetauvcsrc_src_template =
GST_STATIC_PAD_TEMPLATE("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS("video/x-h264, "
	"width = 3840, "
	"height = 1920, "
	"framerate = 30000/1001, "
	"stream-format = (string) byte-stream, "
	"alignment = (string) au, " "profile = (string) constrained-baseline")
    );

/* class initialization */

G_DEFINE_TYPE_WITH_CODE(GstThetauvcsrc, gst_thetauvcsrc, GST_TYPE_PUSH_SRC,
    GST_DEBUG_CATEGORY_INIT
    (gst_thetauvcsrc_debug_category, "thetauvcsrc", 0,
	"debug category for thetauvcsrc element"));

static void
gst_thetauvcsrc_class_init(GstThetauvcsrcClass * klass)
{
    GObjectClass *gobject_class = G_OBJECT_CLASS(klass);
    GstBaseSrcClass *base_src_class = GST_BASE_SRC_CLASS(klass);
    GstPushSrcClass *push_src_class = GST_PUSH_SRC_CLASS(klass);

    /* Setting up pads and setting metadata should be moved to
     * base_class_init if you intend to subclass this class. */
    GstCaps *caps, *c;
    caps = gst_caps_new_simple("video/x-h264",
	"width", G_TYPE_INT, 3840,
	"height", G_TYPE_INT, 1920,
	"framerate", GST_TYPE_FRACTION, 30000, 1001,
	"stream-format", G_TYPE_STRING, "byte-stream",
	"alignment", G_TYPE_STRING, "au",
	"profile", G_TYPE_STRING, "constrained-baseline", NULL);

    c = gst_caps_copy(caps);
    gst_caps_set_simple(c, "width", G_TYPE_INT, 1920, "height", G_TYPE_INT,
	960, NULL);
    gst_caps_append(caps, c);
    gst_element_class_add_pad_template(GST_ELEMENT_CLASS(klass),
	gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, caps));
    /* 
     * gst_element_class_add_static_pad_template (GST_ELEMENT_CLASS (klass),
     * &gst_thetauvcsrc_src_template);
     */

    gst_element_class_set_static_metadata(GST_ELEMENT_CLASS(klass),
	"Theta UVC video source",
	"Generic", "Reads live video from THETA V/Z1", "Koji Takeo <nickel110@icloud.com>");

    gobject_class->set_property = gst_thetauvcsrc_set_property;
    gobject_class->get_property = gst_thetauvcsrc_get_property;
//  gobject_class->dispose = gst_thetauvcsrc_dispose;
    gobject_class->finalize = gst_thetauvcsrc_finalize;
    base_src_class->get_caps = GST_DEBUG_FUNCPTR(gst_thetauvcsrc_get_caps);
    base_src_class->negotiate = GST_DEBUG_FUNCPTR(gst_thetauvcsrc_negotiate);
//  base_src_class->fixate = GST_DEBUG_FUNCPTR (gst_thetauvcsrc_fixate);
    base_src_class->set_caps = GST_DEBUG_FUNCPTR(gst_thetauvcsrc_set_caps);
//  base_src_class->decide_allocation =
//  GST_DEBUG_FUNCPTR (gst_thetauvcsrc_decide_allocation);
    base_src_class->start = GST_DEBUG_FUNCPTR(gst_thetauvcsrc_start);
    base_src_class->stop = GST_DEBUG_FUNCPTR(gst_thetauvcsrc_stop);
//  base_src_class->get_times = GST_DEBUG_FUNCPTR (gst_thetauvcsrc_get_times);
//  base_src_class->get_size = GST_DEBUG_FUNCPTR (gst_thetauvcsrc_get_size);
    base_src_class->is_seekable =
	GST_DEBUG_FUNCPTR(gst_thetauvcsrc_is_seekable);
//  base_src_class->prepare_seek_segment =
//  GST_DEBUG_FUNCPTR (gst_thetauvcsrc_prepare_seek_segment);
//  base_src_class->do_seek = GST_DEBUG_FUNCPTR (gst_thetauvcsrc_do_seek);
//  base_src_class->unlock = GST_DEBUG_FUNCPTR (gst_thetauvcsrc_unlock);
//  base_src_class->unlock_stop = GST_DEBUG_FUNCPTR (gst_thetauvcsrc_unlock_stop);
    base_src_class->query = GST_DEBUG_FUNCPTR(gst_thetauvcsrc_query);
//  base_src_class->event = GST_DEBUG_FUNCPTR (gst_thetauvcsrc_event);
//  base_src_class->create = GST_DEBUG_FUNCPTR (gst_thetauvcsrc_create);
//  base_src_class->alloc = GST_DEBUG_FUNCPTR (gst_thetauvcsrc_alloc);
//  base_src_class->fill = GST_DEBUG_FUNCPTR (gst_thetauvcsrc_fill);

    push_src_class->create = GST_DEBUG_FUNCPTR(gst_thetauvcsrc_create);

    g_object_class_install_property(gobject_class, PROP_HW_SERIAL,
	g_param_spec_string("serial",
	    "Serial number",
	    "The serial number of the THETA", NULL,
	    (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
    g_object_class_install_property(gobject_class, PROP_DEVICE_NUM,
	g_param_spec_int("device-number",
	    "Device number",
	    "Theta device to use", -1, G_MAXINT, -1,
	    (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)));
    g_object_class_install_property(gobject_class, PROP_MODE,
	g_param_spec_enum("mode", "Video mode",
	    "Video mode to playback",
	    gst_thetauvc_mode_get_type(), 0,
	    (G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | G_PARAM_CONSTRUCT)));
    g_object_class_install_property(gobject_class, PROP_DEVICE_INDEX,
	g_param_spec_int("device-index",
	    "Device index",
	    "Index of the opened device", -1, G_MAXINT, -1,
	    (G_PARAM_READABLE | G_PARAM_STATIC_STRINGS)));
}

static void
gst_thetauvcsrc_init(GstThetauvcsrc * thetauvcsrc)
{
    g_mutex_init(&thetauvcsrc->lock);
    g_cond_init(&thetauvcsrc->cond);
    thetauvcsrc->queue =
	gst_queue_array_new_for_struct(sizeof(GstBuffer *), 5);
    thetauvcsrc->ctx = NULL;
    thetauvcsrc->devh = NULL;
}

void
gst_thetauvcsrc_set_property(GObject * object, guint property_id,
    const GValue * value, GParamSpec * pspec)
{
    GstThetauvcsrc *thetauvcsrc = GST_THETAUVCSRC(object);

    GST_DEBUG_OBJECT(thetauvcsrc, "set_property");

    switch (property_id) {
    case PROP_HW_SERIAL:
	thetauvcsrc->serial = g_strdup(g_value_get_string(value));
	break;
    case PROP_DEVICE_NUM:
	thetauvcsrc->device_number = g_value_get_int(value);
	break;
    case PROP_MODE:
	thetauvcsrc->mode = (GstThetauvcModeEnum) g_value_get_enum(value);
	break;
    default:
	G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
	break;
    }
}

void
gst_thetauvcsrc_get_property(GObject * object, guint property_id,
    GValue * value, GParamSpec * pspec)
{
    GstThetauvcsrc *thetauvcsrc = GST_THETAUVCSRC(object);

    GST_DEBUG_OBJECT(thetauvcsrc, "get_property");

    switch (property_id) {
    case PROP_HW_SERIAL:
	g_value_set_string(value, thetauvcsrc->serial);
	break;
    case PROP_DEVICE_NUM:
	g_value_set_int(value, thetauvcsrc->device_number);
	break;
    case PROP_MODE:
	g_value_set_enum(value, thetauvcsrc->mode);
	break;
    case PROP_DEVICE_INDEX:
	g_value_set_int(value, thetauvcsrc->ctx ? thetauvcsrc->device_index : -1);
	break;
    default:
	G_OBJECT_WARN_INVALID_PROPERTY_ID(object, property_id, pspec);
	break;
    }
}

void
gst_thetauvcsrc_dispose(GObject * object)
{
    GstThetauvcsrc *thetauvcsrc = GST_THETAUVCSRC(object);

    GST_DEBUG_OBJECT(thetauvcsrc, "dispose");

    /* clean up as possible.  may be called multiple times */

    G_OBJECT_CLASS(gst_thetauvcsrc_parent_class)->dispose(object);
}

void
gst_thetauvcsrc_finalize(GObject * object)
{
    GstThetauvcsrc *thetauvcsrc = GST_THETAUVCSRC(object);

    GST_DEBUG_OBJECT(thetauvcsrc, "finalize");

    /* clean up object here */
    g_mutex_clear(&thetauvcsrc->lock);
    g_cond_clear(&thetauvcsrc->cond);

    if (thetauvcsrc->queue) {
	while (gst_queue_array_get_length(thetauvcsrc->queue) > 0) {
	    GstBuffer *t;
	    t = (GstBuffer *)
		gst_queue_array_pop_head_struct(thetauvcsrc->queue);
	    gst_buffer_unref(t);
	}
	gst_queue_array_free(thetauvcsrc->queue);
	thetauvcsrc->queue = NULL;
    }

    if (thetauvcsrc->devh) {
	uvc_close(thetauvcsrc->devh);
	uvc_exit(thetauvcsrc->ctx);
    }

    G_OBJECT_CLASS(gst_thetauvcsrc_parent_class)->finalize(object);
}

/* get caps from subclass */
static GstCaps *
gst_thetauvcsrc_get_caps(GstBaseSrc * src, GstCaps * filter)
{
    GstThetauvcsrc *thetauvcsrc = GST_THETAUVCSRC(src);

    GST_DEBUG_OBJECT(thetauvcsrc, "get_caps");

    GstCaps *caps, *rcaps;
    caps = gst_caps_from_string("video/x-h264,framerate=30000/1001,"
	"stream-format=byte-stream,alignment=au");
    GstThetauvcModeEnum mode;

    g_mutex_lock(&thetauvcsrc->lock);
    mode = thetauvcsrc->mode;
    g_mutex_unlock(&thetauvcsrc->lock);
    switch (mode) {
    case GST_THETAUVC_MODE_2K:
	gst_caps_set_simple(caps, "width", G_TYPE_INT, 1920, "height",
	    G_TYPE_INT, 960, NULL);
	break;
    case GST_THETAUVC_MODE_4K:
	gst_caps_set_simple(caps, "width", G_TYPE_INT, 3840, "height",
	    G_TYPE_INT, 1920, NULL);
	break;
    }

    if (filter) {
	rcaps =
	    gst_caps_intersect_full(filter, caps, GST_CAPS_INTERSECT_FIRST);
	gst_caps_unref(caps);
    } else {
	rcaps = caps;
    }
    return rcaps;
}

/* decide on caps */
static  gboolean
gst_thetauvcsrc_negotiate(GstBaseSrc * src)
{
    GstThetauvcsrc *thetauvcsrc = GST_THETAUVCSRC(src);

    GST_DEBUG_OBJECT(thetauvcsrc, "negotiate");

    return TRUE;
}

/* called if, in negotiation, caps need fixating */
static GstCaps *
gst_thetauvcsrc_fixate(GstBaseSrc * src, GstCaps * caps)
{
    GstThetauvcsrc *thetauvcsrc = GST_THETAUVCSRC(src);

    GST_DEBUG_OBJECT(thetauvcsrc, "fixate");

    return NULL;
}

/* notify the subclass of new caps */
static  gboolean
gst_thetauvcsrc_set_caps(GstBaseSrc * src, GstCaps * caps)
{
    GstThetauvcsrc *thetauvcsrc = GST_THETAUVCSRC(src);

    GST_DEBUG_OBJECT(thetauvcsrc, "set_caps");
    GST_DEBUG_OBJECT(thetauvcsrc, "%s", gst_caps_to_string(caps));

    return TRUE;
}

/* setup allocation query */
static  gboolean
gst_thetauvcsrc_decide_allocation(GstBaseSrc * src, GstQuery * query)
{
    GstThetauvcsrc *thetauvcsrc = GST_THETAUVCSRC(src);

    GST_DEBUG_OBJECT(thetauvcsrc, "decide_allocation");

    return TRUE;
}

void
cb(uvc_frame_t * frame, void *ptr)
{
    skipFrameCount--;
    if (skipFrameCount < 0) skipFrameCount = skipFrameNum;
    else return;

    GstThetauvcsrc *thetauvcsrc;
    GstBuffer *buffer;
    GstMapInfo map;
    guint64 interval;

    thetauvcsrc = (GstThetauvcsrc *) ptr;
    buffer = gst_buffer_new_allocate(NULL, frame->data_bytes, NULL);
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, frame->data, frame->data_bytes);
    gst_buffer_unmap(buffer, &map);

    interval = thetauvcsrc->ctrl.dwFrameInterval * 100;
    GST_BUFFER_PTS(buffer) = frame->sequence * interval;
    GST_BUFFER_DTS(buffer) = GST_CLOCK_TIME_NONE;
    GST_BUFFER_DURATION(buffer) = interval;
    GST_BUFFER_OFFSET(buffer) = frame->sequence;
    GST_BUFFER_TIMESTAMP(buffer) = thetauvcsrc->framecount * interval;

    g_mutex_lock(&thetauvcsrc->lock);
    gst_queue_array_push_tail(thetauvcsrc->queue, buffer);
    thetauvcsrc->framecount = 0;
    g_cond_signal(&thetauvcsrc->cond);
    g_mutex_unlock(&thetauvcsrc->lock);

    return;
}

/* start and stop processing, ideal for opening/closing the resource */
static  gboolean
gst_thetauvcsrc_start(GstBaseSrc * src)
{
    GstThetauvcsrc *thetauvcsrc = GST_THETAUVCSRC(src);
    uvc_error_t res;

    GST_DEBUG_OBJECT(thetauvcsrc, "start");
    GST_DEBUG_OBJECT(thetauvcsrc, "dev=%d mode=%d",
	thetauvcsrc->device_number, thetauvcsrc->mode);

    res = uvc_init(&thetauvcsrc->ctx, NULL);
    if (res != UVC_SUCCESS) {
	thetauvcsrc->ctx = NULL;
	GST_ELEMENT_ERROR(src, LIBRARY, INIT,
	    ("Could not initialize libuvc."), (NULL));
	return FALSE;
    }

    if (thetauvcsrc->serial != NULL) {
	res = thetauvc_find_device_by_serial(thetauvcsrc->ctx, &thetauvcsrc->dev,
		thetauvcsrc->serial);
	if (res != UVC_SUCCESS) {
	    GST_ELEMENT_ERROR(src, RESOURCE, NOT_FOUND,
		("Theta (serial:%s) not found.", thetauvcsrc->serial), (NULL));
	    uvc_exit(thetauvcsrc->ctx);
	    return FALSE;
	}
	thetauvcsrc->device_index = -1;
	res = uvc_open(thetauvcsrc->dev, &thetauvcsrc->devh);
    } else {
	if (thetauvcsrc->device_number == -1) {
	    int i;
	    i = 0;
	    while(1) {
		res = thetauvc_find_device(thetauvcsrc->ctx, &thetauvcsrc->dev, i);
		if (res != UVC_SUCCESS)
		    break;
		res = uvc_open(thetauvcsrc->dev, &thetauvcsrc->devh);
		if (res == UVC_SUCCESS) {
		    thetauvcsrc->device_index = i;
		    break;
		}
		uvc_unref_device(thetauvcsrc->dev);
		i++;
	    };

	    if (res != UVC_SUCCESS) {
		if (i == 0)
		    GST_ELEMENT_ERROR(src, RESOURCE, NOT_FOUND,
		   	 ("Theta not found."), (NULL));
		else
		    GST_ELEMENT_ERROR(src, RESOURCE, NOT_FOUND,
		   	 ("Found %d Theta(s), but none available.", i), (NULL));
		uvc_exit(thetauvcsrc->ctx);
		return FALSE;
	    }
	} else {
	    res = thetauvc_find_device(thetauvcsrc->ctx, &thetauvcsrc->dev,
		thetauvcsrc->device_number);
	    if (res != UVC_SUCCESS) {
		GST_ELEMENT_ERROR(src, RESOURCE, NOT_FOUND,
		    ("Theta not found."), (NULL));
		uvc_exit(thetauvcsrc->ctx);
		return FALSE;
	    }
	    res = uvc_open(thetauvcsrc->dev, &thetauvcsrc->devh);
	    thetauvcsrc->device_index = thetauvcsrc->device_number;
	}
    }

    if (res != UVC_SUCCESS) {
	GST_ELEMENT_ERROR(src, RESOURCE, OPEN_READ_WRITE,
	    ("Could not open Theta."), (NULL));
	uvc_exit(thetauvcsrc->ctx);
	return FALSE;
    }

    if (thetauvcsrc->serial == NULL) {
	uvc_device_descriptor_t *desc;
	if (uvc_get_device_descriptor(thetauvcsrc->dev, &desc) == UVC_SUCCESS) {
	    thetauvcsrc->serial = g_strdup(desc->serialNumber);
	    uvc_free_device_descriptor(desc);
	}
    }
    GST_DEBUG_OBJECT(thetauvcsrc, "Serial: %s", thetauvcsrc->serial);

    res = thetauvc_get_stream_ctrl_format_size(thetauvcsrc->devh,
	thetauvcsrc->mode, &thetauvcsrc->ctrl);

    uvc_start_streaming(thetauvcsrc->devh, &thetauvcsrc->ctrl, cb,
	thetauvcsrc, 0);
    thetauvcsrc->framecount = 0;

    return TRUE;
}

static  gboolean
gst_thetauvcsrc_stop(GstBaseSrc * src)
{
    GstThetauvcsrc *thetauvcsrc = GST_THETAUVCSRC(src);

    GST_DEBUG_OBJECT(thetauvcsrc, "stop");

    uvc_stop_streaming(thetauvcsrc->devh);

    return TRUE;
}

/* given a buffer, return start and stop time when it should be pushed
 * out. The base class will sync on the clock using these times. */
static void
gst_thetauvcsrc_get_times(GstBaseSrc * src, GstBuffer * buffer,
    GstClockTime * start, GstClockTime * end)
{
    GstThetauvcsrc *thetauvcsrc = GST_THETAUVCSRC(src);

    GST_DEBUG_OBJECT(thetauvcsrc, "get_times");

}

/* get the total size of the resource in bytes */
static  gboolean
gst_thetauvcsrc_get_size(GstBaseSrc * src, guint64 * size)
{
    GstThetauvcsrc *thetauvcsrc = GST_THETAUVCSRC(src);

    GST_DEBUG_OBJECT(thetauvcsrc, "get_size");

    return TRUE;
}

/* check if the resource is seekable */
static  gboolean
gst_thetauvcsrc_is_seekable(GstBaseSrc * src)
{
    GstThetauvcsrc *thetauvcsrc = GST_THETAUVCSRC(src);

    GST_DEBUG_OBJECT(thetauvcsrc, "is_seekable");

    return FALSE;
}

/* Prepare the segment on which to perform do_seek(), converting to the
 * current basesrc format. */
static  gboolean
gst_thetauvcsrc_prepare_seek_segment(GstBaseSrc * src, GstEvent * seek,
    GstSegment * segment)
{
    GstThetauvcsrc *thetauvcsrc = GST_THETAUVCSRC(src);

    GST_DEBUG_OBJECT(thetauvcsrc, "prepare_seek_segment");

    return TRUE;
}

/* notify subclasses of a seek */
static  gboolean
gst_thetauvcsrc_do_seek(GstBaseSrc * src, GstSegment * segment)
{
    GstThetauvcsrc *thetauvcsrc = GST_THETAUVCSRC(src);

    GST_DEBUG_OBJECT(thetauvcsrc, "do_seek");

    return TRUE;
}

/* unlock any pending access to the resource. subclasses should unlock
 * any function ASAP. */
static  gboolean
gst_thetauvcsrc_unlock(GstBaseSrc * src)
{
    GstThetauvcsrc *thetauvcsrc = GST_THETAUVCSRC(src);

    GST_DEBUG_OBJECT(thetauvcsrc, "unlock");

    return TRUE;
}

/* Clear any pending unlock request, as we succeeded in unlocking */
static  gboolean
gst_thetauvcsrc_unlock_stop(GstBaseSrc * src)
{
    GstThetauvcsrc *thetauvcsrc = GST_THETAUVCSRC(src);

    GST_DEBUG_OBJECT(thetauvcsrc, "unlock_stop");

    return TRUE;
}

/* notify subclasses of a query */
static  gboolean
gst_thetauvcsrc_query(GstBaseSrc * src, GstQuery * query)
{
    GstThetauvcsrc *thetauvcsrc = GST_THETAUVCSRC(src);

    GST_DEBUG_OBJECT(thetauvcsrc, "query");
    switch (GST_QUERY_TYPE(query)) {
    case GST_QUERY_LATENCY:
	{
	    GstClockTime interval;
	    interval = gst_util_uint64_scale_ceil(GST_SECOND, 1001, 30000);
	    gst_query_set_latency(query, TRUE, interval, interval * 8);
	}
	break;
    default:
	GST_BASE_SRC_CLASS(gst_thetauvcsrc_parent_class)->query(src, query);
	break;
    }
    return TRUE;
}

/* notify subclasses of an event */
static  gboolean
gst_thetauvcsrc_event(GstBaseSrc * src, GstEvent * event)
{
    GstThetauvcsrc *thetauvcsrc = GST_THETAUVCSRC(src);

    GST_DEBUG_OBJECT(thetauvcsrc, "event");

    return TRUE;
}

/* ask the subclass to create a buffer with offset and size, the default
 * implementation will call alloc and fill. */
static  GstFlowReturn
gst_thetauvcsrc_create(GstPushSrc * src, GstBuffer ** buf)
{
    GstThetauvcsrc *thetauvcsrc = GST_THETAUVCSRC(src);

    GST_DEBUG_OBJECT(thetauvcsrc, "create");

    g_mutex_lock(&thetauvcsrc->lock);
    while (gst_queue_array_is_empty(thetauvcsrc->queue))
	g_cond_wait(&thetauvcsrc->cond, &thetauvcsrc->lock);

    *buf = (GstBuffer *) gst_queue_array_pop_head(thetauvcsrc->queue);
    GST_DEBUG_OBJECT(thetauvcsrc, "l %lx %d", (unsigned long) *buf,
	(*buf)->mini_object.refcount);
    g_mutex_unlock(&thetauvcsrc->lock);

    return GST_FLOW_OK;
}

/* ask the subclass to allocate an output buffer. The default implementation
 * will use the negotiated allocator. */
static  GstFlowReturn
gst_thetauvcsrc_alloc(GstBaseSrc * src, guint64 offset, guint size,
    GstBuffer ** buf)
{
    GstThetauvcsrc *thetauvcsrc = GST_THETAUVCSRC(src);

    GST_DEBUG_OBJECT(thetauvcsrc, "alloc");

    return GST_FLOW_OK;
}

/* ask the subclass to fill the buffer with data from offset and size */
static  GstFlowReturn
gst_thetauvcsrc_fill(GstBaseSrc * src, guint64 offset, guint size,
    GstBuffer * buf)
{
    GstThetauvcsrc *thetauvcsrc = GST_THETAUVCSRC(src);

    GST_DEBUG_OBJECT(thetauvcsrc, "fill");

    return GST_FLOW_OK;
}

GType
gst_thetauvc_mode_get_type(void)
{
    static gsize id = 0;
    static const GEnumValue mode[] = {
	{GST_THETAUVC_MODE_2K, "1920x960", "2K"},
	{GST_THETAUVC_MODE_4K, "3840x1920", "4K"},
	{0, NULL, NULL}
    };

    if (g_once_init_enter(&id)) {
	GType   tmp = g_enum_register_static("GstThetauvcMode", mode);
	g_once_init_leave(&id, tmp);
    }

    return (GType) id;
}

