function openNav() {
    document.getElementById("mySidebar").style.width = "150px";
    document.getElementById("main").style.marginLeft = "175px";
    document.getElementById("openbtn").style.visibility="hidden";
}
  
function closeNav() {
    document.getElementById("mySidebar").style.width = "0";
    document.getElementById("main").style.marginLeft= "0";
    document.getElementById("openbtn").style.visibility="visible";
}

window.onscroll = function() {scrollFunction()};

function scrollFunction() {
  if (document.body.scrollTop > 50 || document.documentElement.scrollTop > 50) {
    document.getElementById("title").innerHTML='MANUDB';
  } else {
    document.getElementById("title").innerHTML='MANUDB<br>MAmmalian NUclear mitochondrial sequences DataBase';
  }
}