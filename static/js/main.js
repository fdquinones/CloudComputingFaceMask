
(function ($) {
	"use strict";
	$('.column100').on('mouseover',function(){
		var table1 = $(this).parent().parent().parent();
		var table2 = $(this).parent().parent();
		var verTable = $(table1).data('vertable')+"";
		var column = $(this).data('column') + ""; 

		$(table2).find("."+column).addClass('hov-column-'+ verTable);
		$(table1).find(".row100.head ."+column).addClass('hov-column-head-'+ verTable);
	});

	$('.column100').on('mouseout',function(){
		var table1 = $(this).parent().parent().parent();
		var table2 = $(this).parent().parent();
		var verTable = $(table1).data('vertable')+"";
		var column = $(this).data('column') + ""; 

		$(table2).find("."+column).removeClass('hov-column-'+ verTable);
		$(table1).find(".row100.head ."+column).removeClass('hov-column-head-'+ verTable);
	});
	
	// Your web app's Firebase configuration
	var firebaseConfig = {
		apiKey: "AIzaSyBNXxTi201phes4B24moqj_bUIAZvo8cbw",
		authDomain: "utplcovid.firebaseapp.com",
		projectId: "utplcovid",
		storageBucket: "utplcovid.appspot.com",
		messagingSenderId: "1052404064289",
		appId: "1:1052404064289:web:385ce48af2131cc38ec360"
	};
	// Initialize Firebase
	firebase.initializeApp(firebaseConfig);

	var article_id = 1;
	var article = {
		'fecha': Date.now(),
		'cpu': 50,
		'imagen': 'hola.jpg',
	}

	firebase.database().ref('facemask/' + article_id).set(article);
	
	var eventsRef = firebase.database().ref('facemask');
	eventsRef.on('child_added', (data) => {
			console.log("llegando evento en portal de utpl");
			console.log(data.val());
			console.log(data.key());
		//addCommentElement(postElement, data.key, data.val().text, data.val().author);
	});


    

})(jQuery);