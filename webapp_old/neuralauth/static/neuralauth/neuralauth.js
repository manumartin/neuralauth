var neuralauth = {};
(function() { 
 
	console.log('[*] Neuralauth v1.0 started')
	
	this.EVENT_BUFFER_SIZE = 300;

	// user being recorded
	var _user = undefined
	
	// event buffer
	var _events = []

	var _onBatchSentCb = function() {}
	var _inputRecordCb = function() {}
    var _onRecognizeCb = function() {}

	/* Only write methods require CSRF protection */
	function _csrfSafeMethod(method) {
	    // these HTTP methods do not require CSRF protection
	    return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
	}

	/* Reads a Cookie, TODO: use the js-cookie library for this */
	function _getCookie(name) {
	    var cookieValue = null;
	    if (document.cookie && document.cookie !== '') {
	        var cookies = document.cookie.split(';');
	        for (var i = 0; i < cookies.length; i++) {
	            var cookie = jQuery.trim(cookies[i]);
	            // Does this cookie string begin with the name we want?
	            if (cookie.substring(0, name.length + 1) === (name + '=')) {
	                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
	                break;
	            }
	        }
	    }
	    return cookieValue;
	}

	/* Setup CSRFToken for unsafe HTTP methods if they arent cross-domain */
	$.ajaxSetup({
	    beforeSend: function(xhr, settings) {
	        if (!_csrfSafeMethod(settings.type) && !this.crossDomain) {
	            xhr.setRequestHeader("X-CSRFToken", _getCookie('csrftoken'));
	        }
	    }
	});

	/* Flushes the input buffer and sends the data to the neural network for processing */
	function _record(event) {

		if(_events.length < neuralauth.EVENT_BUFFER_SIZE) {
			_events.push(event);
		} else {
			console.log('[*] Sending user input to neural network...');
			_onBatchSentCb();
			$.ajax({
				method: "POST",
				url: "/neuralauth/recognize/",
				data: JSON.stringify({user: _user, input:_events}),
				success: function(data) {
                    _onRecognizeCb(data);
					console.log('[*] user input received:');
					console.log(data);	
				},
				contentType: "application/json",
				dataType:"json"
			});
			_events = []
		}

	}

	this.getUser = function() {
		return _user;
	}

	/* Calls the compare service, will return a comparison of every recorded
	user with every other one */
	this.compare = function(callback) {

		console.log('[*] User comparison requested..');

		$.ajax({
			method: "GET",
			url: "/neuralauth/compare",
			success: callback,
			contentType: "application/json",
			dataType:"json"
		});	
	}

	/* Starts to record the user input */
	this.start = function(user) {

		console.log('[*] Started recording user '+user+'.. ');

		_user = user

		$(document).off('mousemove.neuralauth');
		$(document).off('mousedown.neuralauth');
		$(document).off('mousedown.neuralauth');

		$(document).on('mousemove.neuralauth', function(e) {
			console.log('ehe');
			_inputRecordCb(_events.length);
			console.log('done');
			var state;
			if(e.buttons == 0) {
				state = 0; // move
			} else {
				state = 3; // drag
			}

			_record([Date.now(), e.buttons, state, e.screenX, e.screenY]);
		});

		$(document).on('mousedown.neuralauth', function(e) {
			_inputRecordCb(_events.length);
			_record([Date.now(), e.buttons, 1, e.screenX, e.screenY]);
		});

		$(document).on('mouseup.neuralauth', function(e) {
			_inputRecordCb(_events.length);
			_record([Date.now(), e.buttons, 2, e.screenX, e.screenY]);
		});
	}

	this.reset = function() {
		$.ajax({
			method: "POST",
			url: "/neuralauth/reset", 
			success: function(data) {
				console.log('[*] user data resetted.');
			},
			contentType: "application/json",
			dataType:"json"
		});
	}

	/* Stops recording the user input */
	this.stop = function() {
		_events = []
		console.log('[*] Stopped recording user '+_user+'.. ');

		$(document).off('mousemove.neuralauth');
		$(document).off('mousedown.neuralauth');
		$(document).off('mousedown.neuralauth');		
	}

	this.setInputRecordCb = function(callback) {
		_inputRecordCb = callback;
	}

	this.setOnBatchSentCb = function(callback) {
		_onBatchSentCb = callback;
	}
    
    this.setOnRecognizeCb = function(callback) {
        _onRecognizeCb = callback;
    }

}).apply(neuralauth); 


















